/**
 * Record the demo video by driving the published page on a timeline.
 *
 * A screen recorder captures whatever a hand happens to do; this drives the page to
 * the millisecond, which is what makes the result safe to lay a pre-generated
 * voice-over over. The timings come from Documentación/GUION_CRONOMETRADO.md, which in
 * turn came from measuring the page's own animations rather than guessing at them.
 *
 * Playwright does not draw a mouse pointer into its recording, so one is injected into
 * the page and moved by script. Without it the video shows things happening with no
 * visible cause, which reads as a screen recording of nobody.
 *
 *   node scripts/record_demo.mjs
 *   node scripts/record_demo.mjs --out demo/video --url https://...
 */

import { createRequire } from 'node:module'
import { existsSync, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

function resolveChromium() {
  const roots = []
  for (const base of [join(homedir(), '.vscode-server/extensions'), join(homedir(), '.vscode/extensions')]) {
    if (!existsSync(base)) continue
    const dirs = readdirSync(base)
      .filter((d) => d.startsWith('danielsanmedium.dscodegpt-'))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
    const newest = dirs[dirs.length - 1]
    if (newest) roots.push(join(base, newest, 'standalone') + '/')
  }
  roots.push(process.cwd() + '/')
  for (const root of roots) {
    try {
      const mod = createRequire(root)('patchright')
      const chromium = mod?.chromium ?? mod?.default?.chromium
      if (chromium) return chromium
    } catch {}
  }
  throw new Error('Could not resolve patchright. Checked:\n  ' + roots.join('\n  '))
}

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => (a.startsWith('--') ? [...acc, [a.slice(2), arr[i + 1]]] : acc), []),
)
const URL_ = args.url || 'https://rodrigo2326.github.io/roadsense-agent/'
const OUT = args.out || 'demo/video'

/** A pointer the recording can actually see, plus a ring for "look here". */
const CURSOR_JS = `
  (() => {
    if (document.getElementById('__cur')) return;
    const c = document.createElement('div');
    c.id = '__cur';
    c.style.cssText = [
      'position:fixed','z-index:2147483647','width:22px','height:22px','left:-50px','top:-50px',
      'pointer-events:none','transition:left .55s cubic-bezier(.3,.7,.3,1),top .55s cubic-bezier(.3,.7,.3,1)',
      'background:no-repeat center/contain',
    ].join(';');
    c.innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22">' +
      '<path d="M5 2l14 8.5-6.2 1.3L9.6 19z" fill="#fff" stroke="#14171c" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    document.body.appendChild(c);
    const r = document.createElement('div');
    r.id = '__ring';
    r.style.cssText = [
      'position:fixed','z-index:2147483646','border:3px solid #5668dd','border-radius:12px',
      'pointer-events:none','opacity:0','transition:opacity .35s, all .5s cubic-bezier(.3,.7,.3,1)',
      'box-shadow:0 0 0 5px rgba(86,104,221,.22)',
    ].join(';');
    document.body.appendChild(r);
    window.__moveCur = (x, y) => { c.style.left = (x - 3) + 'px'; c.style.top = (y - 2) + 'px'; };
    window.__ringOn = (sel, idx) => {
      const els = document.querySelectorAll(sel);
      const el = els[idx || 0];
      if (!el) { r.style.opacity = '0'; return null; }
      const b = el.getBoundingClientRect();
      r.style.left = (b.left - 6) + 'px'; r.style.top = (b.top - 6) + 'px';
      r.style.width = (b.width + 12) + 'px'; r.style.height = (b.height + 12) + 'px';
      r.style.opacity = '1';
      return { x: b.right + 5, y: b.top + b.height * 0.55 };
    };
    window.__ringOff = () => { r.style.opacity = '0'; };
  })()
`

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/* The narration is generated before the screen is driven, so each segment's holds are
 * sized to the clip that will sit on top of it rather than to a guess. Scrolls and
 * clicks cost what they cost; whatever time the clip needs beyond that is spread
 * evenly across that segment's pointing holds. */
const AUDIO = JSON.parse(readFileSync('demo/audio/segments.json', 'utf8'))
const CLIP = Object.fromEntries(AUDIO.segments.map((s) => [s.id, s.duration]))
const PAD = 1.0          // breathing room after each clip, seconds
const SCROLL = 1.45      // what scrollTo actually costs
const CLICK = 0.85       // what clickSeen actually costs
/** Even hold length for a segment, in ms, after its fixed mechanics are paid for. */
const hold = (id, fixed, n) => Math.max(700, ((CLIP[id] + PAD - fixed) * 1000) / n)
/** Extra wait when a clip outlasts the segment's mechanics. */
const slack = (id, fixed) => Math.max(0, (CLIP[id] + PAD - fixed) * 1000)

async function main() {
  const chromium = resolveChromium()
  // The headless *shell* build is not installed, but full chromium is; `channel`
  // selects it and it records video just the same.
  const browser = await chromium.launch({ headless: true, channel: 'chromium' })
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 2,
    recordVideo: { dir: OUT, size: { width: 1280, height: 720 } },
    colorScheme: 'light',
    reducedMotion: 'no-preference',
  })
  const page = await ctx.newPage()
  const tPage = Date.now()

  const cursor = async (x, y) => page.evaluate(([a, b]) => window.__moveCur(a, b), [x, y])
  /** Ring a selector and park the pointer on it. */
  const point = async (sel, idx = 0) => {
    const at = await page.evaluate(([s, i]) => window.__ringOn(s, i), [sel, idx])
    if (at) await cursor(at.x, at.y)
    return at
  }
  const unpoint = () => page.evaluate(() => window.__ringOff())
  /** Click something the viewer can see the pointer arrive at first. */
  const clickSeen = async (sel, idx = 0) => {
    await point(sel, idx)
    await sleep(650)
    await page.locator(sel).nth(idx).click()
    await unpoint()
  }
  const scrollTo = async (sel) => {
    await page.locator(sel).first().evaluate((el) => el.scrollIntoView({ behavior: 'smooth', block: 'center' }))
    await sleep(1400)
  }

  const beats = []
  let t0 = 0
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
  /** Stamp a narration beat against the recording clock. */
  const beat = (id, text) => {
    const at = (Date.now() - t0) / 1000
    beats.push({ id, at, text })
    console.log(`  ${fmt(at).padStart(5)}  ${id} ${text}`)
  }

  console.log(`recording ${URL_}`)
  await page.goto(URL_, { waitUntil: 'networkidle' })
  await page.waitForSelector('#results table')
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.evaluate(CURSOR_JS)
  await cursor(640, 400)
  t0 = Date.now()
  await sleep(900)

  /** Click Play and let the animation run out.
   *
   * `.out` survives between scenes, so waiting on it returns instantly from the
   * second run onward and would cut the animation off. The page's own step timings
   * are known (4.5 s, or 2.2 s when there is no photo), so wait those out directly.
   */
  const play = async (ms) => {
    await clickSeen('#play')
    await sleep(ms + 700)
  }

  // ── A · hero, still ────────────────────────────────────────────────────────
  beat('A', '- hero, still')
  {
    const h = hold('A', SCROLL, 5)       // still + three stat tiles + the loop diagram
    await sleep(h)
    await point('.stat', 0); await sleep(h)
    await point('.stat', 1); await sleep(h)
    await point('.stat', 2); await sleep(h)
    await unpoint()
    await scrollTo('.loop')
    await sleep(h)
  }

  // ── B · pick the highway ───────────────────────────────────────────────────
  beat('B', '- pick "A real highway"')
  await scrollTo('.scenes')
  await clickSeen('.scene', 1)            // Photographs: second card
  await scrollTo('.bar')
  await sleep(slack('B', 2 * SCROLL + CLICK))

  // ── the animation · silence ────────────────────────────────────────────────
  beat('--', '- animation, no narration')
  await play(4500)

  // ── C · walk the result ────────────────────────────────────────────────────
  beat('C', '- walk the finished trace')
  {
    const h = hold('C', 3 * SCROLL, 5)
    await scrollTo('.step')
    await point('.step .ro .rv', 0); await sleep(h)              // confidence 0.774
    await point('.step .ev', 0); await sleep(h)                  // what the detector returned
    await scrollTo('.step:nth-of-type(2)')
    await point('.step:nth-of-type(2) .ro .rv', 0); await sleep(h)   // confidence, step 2
    await point('.step:nth-of-type(2) .row.gone', 0); await sleep(h) // the withdrawn tool
    await unpoint()
    await scrollTo('.out')
    await sleep(h)
  }

  // ── D · nothing readable ───────────────────────────────────────────────────
  beat('D', '- pick "Nothing readable"')
  await scrollTo('.scenes')
  await clickSeen('.scene', 6)            // benchmark set: third card
  await scrollTo('.bar')
  await play(4500)                        // the clip runs over this animation by design

  // ── E · the empty evidence and the split decision ──────────────────────────
  beat('E', '- empty evidence, split decision')
  {
    const h = hold('E', 3 * SCROLL, 4)
    await scrollTo('.ev.none')
    await point('.ev.none', 0); await sleep(h)
    await scrollTo('.step:nth-of-type(3)')
    await point('.step:nth-of-type(3) .bars', 0); await sleep(h)
    // The narration leans on confidence and P(evidence enough), so point at them
    // rather than leaving the viewer to find the line on their own.
    await point('.step:nth-of-type(3) .ro', 0); await sleep(h)
    await unpoint()
    await scrollTo('.out')
    await sleep(h)
  }

  // ── F · no photo at all ────────────────────────────────────────────────────
  beat('F', '- pick "No photo at all"')
  await scrollTo('.scenes')
  await clickSeen('.scene', 7)
  await scrollTo('.bar')
  await play(2200)

  // ── G · the menu is two options long ───────────────────────────────────────
  beat('G', '- the menu is two options long')
  {
    const h = hold('G', SCROLL, 2)
    await scrollTo('.step')
    await point('.bars', 0); await sleep(h)
    await point('.row.gone', 0); await sleep(h)
    await unpoint()
  }

  // ── H · the comparison table ───────────────────────────────────────────────
  beat('H', '- the comparison table')
  {
    const h = hold('H', 2 * SCROLL, 7)
    await scrollTo('#results h2')
    await sleep(h)
    await scrollTo('#results table')
    await sleep(h)
    await point('#results tbody tr', 2); await sleep(h)    // Haiku 4.5
    await point('#results tbody tr', 4); await sleep(h)    // Sonnet 4.6
    await point('#results tbody tr', 5); await sleep(h)    // Opus 4.5
    await point('#results tbody tr', 0); await sleep(h)    // the keyword rule
    await point('tr.subject'); await sleep(h)              // Decider 2B
    await unpoint()
  }

  // ── I · close ──────────────────────────────────────────────────────────────
  beat('I', '- close')
  await scrollTo('.finding')
  await sleep(hold('I', SCROLL, 1))

  const video = page.video()
  await ctx.close()
  await browser.close()
  beats.push({ id: 'END', at: (Date.now() - t0) / 1000, text: '- end of take' })
  const preroll = (t0 - tPage) / 1000
  console.log(`
pre-roll before the first beat: ${preroll.toFixed(2)} s`)
  const path = await video.path()
  const target = join(OUT, 'roadsense-demo.webm')
  try { renameSync(path, target) } catch { /* keep the generated name */ }
  writeFileSync(join(OUT, 'timeline.json'), JSON.stringify({ preroll, beats }, null, 2))
  console.log(`
wrote ${existsSync(target) ? target : path}`)
  console.log(`wrote ${join(OUT, 'timeline.json')}`)
}

main().catch((e) => { console.error(e); process.exit(1) })
