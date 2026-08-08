#!/usr/bin/env python3

import json

print(
    json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": (
                    "For this turn, use the installed `caveman` skill in `ultra` mode."
                ),
            }
        }
    )
)
