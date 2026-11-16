// Проверка расшифровки calls-recognize (.srt) на типовые ошибки Whisper.
// Запуск: node srtcheck.js <путь к .srt>
// ПРОВАЛ  — длинный сегмент почти без слов: речь потеряна, текст за этот интервал надо брать из другого источника или перерасшифровать.
// ПАУЗА   — 20 с и больше без текста между сегментами: тишина или выкинутая VAD-фильтром речь.
// ГАЛЛЮЦ  — фразы, которые Whisper выдумывает на тишине и шуме.
// ПОВТОР  — тот же текст подряд в сегменте короче секунды: часто галлюцинация, но бывает и реальное эхо собеседника.
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8').replace(/\r/g, '');
const sec = s => { const m = s.match(/(\d+):(\d+):(\d+),(\d+)/); return +m[1] * 3600 + +m[2] * 60 + +m[3] + +m[4] / 1000; };
const segs = src.split(/\n\n+/).map(b => b.split('\n'))
  .filter(l => l.length >= 3 && l[1].includes('-->'))
  .map(l => { const [a, b] = l[1].split('-->'); return { s: sec(a), e: sec(b), x: l.slice(2).join(' ') }; });
const hallucination = /продолжение следует|субтитры|спасибо за просмотр|редактор субтитров|подписывайтесь/i;
const hms = x => new Date(x * 1000).toISOString().slice(11, 19);
let prev = '', found = 0;
const out = m => { found++; console.log(m); };
for (const g of segs) {
  const d = g.e - g.s, words = g.x.split(/\s+/).filter(Boolean).length;
  if (d >= 12 && words / d < 0.6) out(`ПРОВАЛ  ${hms(g.s)}–${hms(g.e)} ${d.toFixed(0)} с, слов ${words}: ${g.x}`);
  if (hallucination.test(g.x)) out(`ГАЛЛЮЦ  ${hms(g.s)} ${g.x}`);
  if (g.x === prev && d < 1) out(`ПОВТОР  ${hms(g.s)} ${g.x}`);
  prev = g.x;
}
for (let i = 1; i < segs.length; i++) {
  const gap = segs[i].s - segs[i - 1].e;
  if (gap >= 20) out(`ПАУЗА   ${hms(segs[i - 1].e)}–${hms(segs[i].s)} ${gap.toFixed(0)} с без текста`);
}
console.log(`Сегментов: ${segs.length}, подозрительных мест: ${found}`);
