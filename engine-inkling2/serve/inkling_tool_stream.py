"""Frame the arguments object in Inkling's native name-first tool envelope."""

import json
import re


_WS = r"[ \t\r\n]*"
_HEAD = re.compile(rf'{_WS}\{{{_WS}"name"{_WS}:{_WS}("(?:[^"\\]|\\.)*"){_WS},{_WS}"args"{_WS}:{_WS}(?=\{{)')
_HEAD_LIMIT = 4096
_CHUNK_SIZE = 256


def _invalid_constant(value):
    raise ValueError(f"{value} is not a JSON constant")


class CanonicalToolStream:
    """Emit fragments of a declared tool's JSON object, withholding its final brace."""

    def __init__(self, names):
        self.names = names
        self.name = None
        self.started = False
        self.closed = False
        self._head = "" if names else None
        self._body = []
        self._depth = 0
        self._quoted = self._escaped = False
        self._pending = []
        self._size = 0

    def feed(self, text):
        if not text:
            return None
        if self.name is None:
            if self._head is None:
                return None
            self._head += text
            match = _HEAD.match(self._head)
            if not match or match.end() >= _HEAD_LIMIT:
                if len(self._head) >= _HEAD_LIMIT:
                    self._head = None
                return None
            text, self._head = self._head[match.end():], None
            try:
                name = json.loads(match[1])
            except ValueError:
                return None
            if name not in self.names:
                return None
            self.name = name
        self._body.append(text)
        if self.closed:
            return None
        end = len(text)
        for i, char in enumerate(text):
            if self._escaped:
                self._escaped = False
            elif self._quoted:
                if char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._quoted = False
            elif char == '"':
                self._quoted = True
            elif char == "{":
                self._depth += 1
            elif char == "}":
                self._depth -= 1
                if self._depth == 0:
                    self.closed, end = True, i
                    break
        if end:
            self._pending.append(text[:end])
            self._size += end
        if self._size < _CHUNK_SIZE:
            return None
        self.started = True
        fragment = "".join(self._pending)
        self._pending.clear()
        self._size = 0
        return fragment

    def finish(self):
        """Call after the model closes the channel. Return the withheld suffix and decoded arguments."""
        body = "".join(self._body)
        try:
            arguments, end = json.JSONDecoder(parse_constant=_invalid_constant).raw_decode(body)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid streamed Inkling tool arguments") from exc
        if not self.closed or not isinstance(arguments, dict) or body[end:].strip(" \t\r\n") != "}":
            raise ValueError("invalid streamed Inkling tool envelope")
        return "".join(self._pending) + "}", arguments
