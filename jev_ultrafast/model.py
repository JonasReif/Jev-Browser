"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import re
import time
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=5)
REQUEST_WALL_TIMEOUT = 15


def _bounded_post(url, key, body):
    """Close a streamed response before retrying if its body exceeds the deadline."""
    deadline = time.monotonic() + REQUEST_WALL_TIMEOUT
    with CLIENT.stream("POST", url, json=body,
                       headers={"Authorization": f"Bearer {key}"}) as response:
        if response.is_error:
            return response.status_code, None
        chunks = []
        for chunk in response.iter_bytes():
            if time.monotonic() >= deadline:
                raise TimeoutError("Model response exceeded the body deadline")
            chunks.append(chunk)
        return response.status_code, json.loads(b"".join(chunks))


def post_json(url, key, body):
    for attempt in range(3):
        try:
            status, result = _bounded_post(url, key, body)
        except (TimeoutError, httpx.TimeoutException):
            if attempt < 2:
                continue
            raise RuntimeError("Model request timed out; no action executed.") from None
        except json.JSONDecodeError:
            if attempt < 2:
                continue
            raise RuntimeError("Model provider returned invalid JSON; no action executed.") from None
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if status in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if status >= 400:
            raise RuntimeError(f"Model provider returned HTTP {status}; no action executed.")
        if not isinstance(result, dict):
            if attempt < 2:
                continue
            raise RuntimeError("Model provider returned an invalid JSON envelope; no action executed.")
        return {**result, "_jev_transport": {"requests": attempt + 1, "usage_unknown": attempt}}
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT",
                  "press_enter": "PRESS_ENTER"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def policy_actions(state):
    """Hide search controls that only repeat or clear the displayed results."""
    actions = state["actions"]
    parsed = urlsplit(state["url"])
    search_page = parsed.path.rstrip("/").split("/")[-1].casefold() == "search"
    query = parse_qs(parsed.query)
    submitted = {value.strip().casefold() for key, values in query.items()
                 if key.casefold() in {"q", "query", "search"} for value in values if value.strip()}
    fields = [a for a in actions if a["kind"] == "fill"]
    same_query = bool(submitted) and any(str(a.get("value", "")).strip().casefold() in submitted
                                         for a in fields)
    empty_search = bool(search_page and fields) and all(not str(a.get("value", "")).strip()
                                                         for a in fields)

    def redundant_search_link(action):
        if not (search_page and action.get("role") == "link" and
                action.get("label", "").strip().casefold() == "search" and action.get("href")):
            return False
        target = urlsplit(urljoin(state["url"], action["href"]))
        return target.path == parsed.path and target.query in {"", parsed.query}

    return [a for a in actions if not (
        (a["kind"] == "press_enter" and
         (same_query and str(a.get("value", "")).strip().casefold() in submitted or empty_search)) or
        (a["kind"] == "click" and
         (redundant_search_link(a) or
          (a.get("role") == "button" and (same_query or empty_search) and
           re.match(r"^(?:search\b|submit\b|go$|find$)", a.get("label", ""), re.I))))
    )]


def choose(state, goal, history):
    elements, targets, controls = action_space(policy_actions(state))
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
        "PRESS_ENTER": "Submit a populated search field by pressing Enter in that field.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    usage_attempts = []
    for attempt in range(2):
        result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
        transport = result.get("_jev_transport", {}) if isinstance(result, dict) else {}
        usage_attempts.extend([None] * (transport.get("requests", 1) - 1))
        usage_attempts.append(result.get("usage") if isinstance(result, dict) else None)
        try:
            answers = result.get("answers") if isinstance(result, dict) else None
            if not isinstance(answers, dict):
                answers = {}
            operation_answer = validate_choice(answers.get("operation", {}), operations)
            operation = operation_answer["choice"]
            target = None
            target_answer = None
            probabilities = {}
            if operation in targets:
                # Unused target heads cannot cause an action. Validate the head selected by the operation.
                target_answer = validate_choice(
                    answers.get(operation.lower() + "_target", {}), targets[operation]
                )
                target = target_answer["choice"]
                choice = targets[operation][target]["id"]
                probabilities = {a["id"]: target_answer["probabilities"][index]
                                 for index, a in targets[operation].items()}
            else:
                choice = controls[operation]["id"] if operation in controls else operation
                probabilities[choice] = operation_answer["probabilities"][operation]
            break
        except ValueError:
            if attempt:
                raise
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "usage_attempts": usage_attempts,
        "request_count": len(usage_attempts),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "kind", "text")}
                           for h in history[-6:]],
    }


def parse_text_reply(content):
    """Accept one JSON value, including a provider's stray Markdown fence."""
    if not isinstance(content, str) or not content.strip():
        return None
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, count=1, flags=re.I)
    try:
        output, end = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        return None
    return output if candidate[end:].strip() in {"", "```"} else None


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    field = context.get("field") or {}
    is_search = field.get("role") == "searchbox" or bool(
        re.search(r"\bsearch\b", field.get("label") or "", re.I)
    )
    history = context.get("recent_actions", [])
    attempted = set()
    if is_search:
        for index, action in enumerate(history):
            if action.get("kind") != "fill" or not action.get("text"):
                continue
            if any(later.get("kind") == "press_enter" or
                   re.search(r"\b(?:search|submit)\b", later.get("action") or "", re.I)
                   for later in history[index + 1:]):
                attempted.add(action["text"].strip().casefold())
    started = time.perf_counter()
    request = {
        "model": model,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        **reasoning,
        "messages": [
            {"role": "system", "content": TEXT_VALUE},
            {"role": "user", "content": json.dumps(context)},
        ],
    }
    usage_attempts = []
    for attempt in range(2):
        result = post_json(base + "/chat/completions", key, request)
        transport = result.get("_jev_transport", {})
        usage_attempts.extend([None] * (transport.get("requests", 1) - 1))
        usage_attempts.append(result.get("usage"))
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        output = parse_text_reply(content)
        if output is not None:
            if is_search and attempted and isinstance(output, dict) and isinstance(output.get("text"), str):
                proposed = re.sub(
                    r"(?:\s+(?:reference|documentation|article|page|interface|repository))+$",
                    "", output["text"].strip(), flags=re.I,
                )
                if len(proposed.split()) == 1 and proposed.casefold() in attempted:
                    if attempt:
                        raise ValueError("Text helper repeated a submitted search; nothing typed.") from None
                    continue
            if output is not None and not (
                attempt == 0 and is_search and attempted and
                isinstance(output, dict) and output.get("text", False) is None
            ):
                break
        if attempt:
            raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    try:
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    if is_search:
        subject = re.sub(r"(?:\s+(?:reference|documentation|article|page|interface|repository))+$",
                         "", value.strip(), flags=re.I)
        if subject:
            value = subject
        if value.casefold() in attempted:
            words = value.split()
            while len(words) > 1:
                words.pop(0)
                candidate = " ".join(words)
                if candidate.casefold() not in attempted:
                    value = candidate
                    break
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
        "usage_attempts": usage_attempts,
        "request_count": len(usage_attempts),
    }
