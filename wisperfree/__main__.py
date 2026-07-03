"""Entry point: ``wisperfree`` (or ``python -m wisperfree``).

Starts the daemon (pipeline + global hotkey) and the localhost API the
tray app connects to.
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="wisperfree")
    parser.add_argument("--host", default=None, help="API bind host")
    parser.add_argument("--port", type=int, default=None, help="API bind port")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import uvicorn

    from wisperfree.api import create_app
    from wisperfree.daemon import Daemon

    daemon = Daemon()
    daemon.start()
    app = create_app(daemon)
    try:
        uvicorn.run(
            app,
            host=args.host or daemon.config.server.host,
            port=args.port or daemon.config.server.port,
            log_level="warning",
        )
    finally:
        daemon.shutdown()


if __name__ == "__main__":
    main()
