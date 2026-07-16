from __future__ import annotations

import platform
import subprocess


def notify(title: str, message: str) -> None:
    if platform.system() != "Darwin":
        return
    safe_title = title.replace('"', "'")
    safe_message = message.replace('"', "'")
    subprocess.run(
        ["osascript", "-e", f'display notification "{safe_message}" with title "{safe_title}"'],
        check=False,
        capture_output=True,
    )
