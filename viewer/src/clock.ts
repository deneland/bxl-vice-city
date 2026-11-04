// Time of day in seconds (0 = midnight, Brussels local time) and calendar days as 'YYYY-MM-DD'.
export const H = 3600, M = 60;

/** 'hh:mm' of a time of day in seconds */
export const hhmm = (t: number) => `${String(Math.floor(t / H) % 24).padStart(2, '0')}:${String(Math.floor(t / M) % 60).padStart(2, '0')}`;

const weekdays = new Map<string, number>();
/** day of the week of a date 'YYYY-MM-DD', Monday first: 0 Monday … 6 Sunday (cached: called every frame) */
export function weekday(date: string) {
  let wd = weekdays.get(date);
  if (wd === undefined) {
    const [y, m, d] = date.split('-').map(Number);
    wd = (new Date(Date.UTC(y, m - 1, d)).getUTCDay() + 6) % 7;
    weekdays.set(date, wd);
  }
  return wd;
}

/** [YYYY-MM-DD, fractional hours] now on the Brussels clock */
export function brusselsNow() {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-CA', { timeZone: 'Europe/Brussels', hourCycle: 'h23',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
    .formatToParts(new Date()).map((x) => [x.type, x.value]));
  return [`${p.year}-${p.month}-${p.day}`, Number(p.hour) + Number(p.minute) / 60 + Number(p.second) / 3600] as const;
}

/** minutes Brussels is ahead of UTC at an instant (CET / CEST) */
function brusselsOffset(d: Date) {
  const tz = new Intl.DateTimeFormat('en', { timeZone: 'Europe/Brussels', timeZoneName: 'shortOffset' })
    .formatToParts(d).find((p) => p.type === 'timeZoneName')?.value ?? 'GMT+1';
  const m = tz.match(/GMT([+-]\d+)(?::(\d+))?/);
  return m ? Number(m[1]) * 60 + Math.sign(Number(m[1])) * Number(m[2] ?? 0) : 60;
}

/** the instant of a Brussels date and time of day (fractional hours) */
export function brusselsInstant(date: string, hours: number) {
  const [y, mo, d] = date.split('-').map(Number);
  const t = new Date(Date.UTC(y, mo - 1, d) + hours * 3600e3);
  return new Date(t.getTime() - brusselsOffset(t) * 60e3);
}
