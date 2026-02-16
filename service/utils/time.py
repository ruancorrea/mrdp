from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

class Time:
    '''
    A class to handle time-related operations, including timezone-aware conversions.
    '''
    DEFAULT_TZ_NAME = "America/Sao_Paulo"

    def __init__(self, tz_name: str = None) -> None:
        '''
        Initializes the Time object with a specific timezone.
        :param tz_name: The name of the timezone (e.g., 'America/Sao_Paulo').
                        Defaults to DEFAULT_TZ_NAME.
        '''
        self.MINUTES_PER_HOUR = 60.0
        self.tz_name = tz_name if tz_name else self.DEFAULT_TZ_NAME
        self.tz = timezone.utc
        if ZoneInfo:
            try:
                self.tz = ZoneInfo(self.tz_name)
            except Exception:
                self.tz = timezone.utc

    def to_timestamp_seconds(self, dt: datetime) -> float:
        '''
        Converts a datetime object to a POSIX timestamp in seconds.
        If the datetime is naive, it's assumed to be in the instance's timezone.
        '''
        if dt is None:
            raise ValueError("datetime é None")
        if dt.tzinfo is not None:
            return dt.timestamp()

        dt_loc = dt.replace(tzinfo=self.tz)
        return dt_loc.timestamp()

    def datetimes_map_to_minutes(self, P_dt_map: dict, T_dt_map: dict) -> tuple[dict, dict, float]:
        '''
        Recebe dicionários/arrays-like de datetimes P_dt_map[node] e T_dt_map[node].
        Retorna dois dicionários com valores em minutos (float) relativos a uma referência (min timestamp).
        Também retorna a referência timestamp (segundos) usada.
        '''
        all_ts = []
        P_ts = {}
        T_ts = {}
        for k, dt in P_dt_map.items():
            ts = self.to_timestamp_seconds(dt)
            P_ts[k] = ts
            all_ts.append(ts)
        for k, dt in T_dt_map.items():
            ts = self.to_timestamp_seconds(dt)
            T_ts[k] = ts
            all_ts.append(ts)
        if not all_ts:
            raise ValueError("Nenhum datetime fornecido")
        ref_ts = min(all_ts)  # referência para zero (poderia ser min(P) ou min de todos)
        # converter para minutos relativos à ref_ts
        P_min = {k: (v - ref_ts) / self.MINUTES_PER_HOUR for k, v in P_ts.items()}
        T_min = {k: (v - ref_ts) / self.MINUTES_PER_HOUR  for k, v in T_ts.items()}
        return P_min, T_min, ref_ts

    def minutes_to_datetime(self, minutes: float, ref_ts: float) -> datetime:
        '''
        Converts a time in minutes (relative to a reference timestamp) back to an aware datetime object
        in the instance's timezone.
        '''
        ts = ref_ts + minutes * self.MINUTES_PER_HOUR
        utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return utc_dt.astimezone(self.tz)

