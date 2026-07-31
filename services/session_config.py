from dataclasses import dataclass


@dataclass(slots=True)
class SessionConfig:
    youtube_url: str
    start_time: float
    end_time: float


class SessionConfigurator:

    DEFAULT_URL = "https://www.youtube.com/watch?v=OwiQgAMqyUs"
    DEFAULT_START = 0.0
    DEFAULT_END = 60.0

    def prompt(self) -> SessionConfig:

        print("\n" + "=" * 60)
        print("Basketball AI - Video Configuration")
        print("=" * 60)

        url = input(
            f"Enter YouTube URL\n[ENTER = {self.DEFAULT_URL}]\n> "
        ).strip()

        if not url:
            url = self.DEFAULT_URL

        start = self._ask_float(
            "Enter START time (seconds)",
            self.DEFAULT_START,
        )

        end = self._ask_float(
            "Enter END time (seconds)",
            self.DEFAULT_END,
        )

        print("=" * 60)

        return SessionConfig(
            youtube_url=url,
            start_time=start,
            end_time=end,
        )

    @staticmethod
    def _ask_float(text: str, default: float) -> float:

        value = input(f"{text} [ENTER = {default}]\n> ").strip()

        try:
            return float(value) if value else default

        except ValueError:

            print("Invalid value. Using default.")

            return default