# Refactor Vision — bamp.codes v2

Drafted 2026-04-28. Successor to the perf audit (`PERFORMANCE_AUDIT.md`).
This document captures the creative vision, the inspiration corpus, the
tech research, and a phased implementation plan.

## 1. Creative vision

### Mood board

A scroll-driven cinematic descent through a **futuristic cyberpunk
megacity**. Visual references:

- **Cyberpunk: Edgerunners** (anime) — neon density, color saturation,
  hand-painted-feeling grime under the gloss.
- **Cyberpunk 2077** (game) — Night City verticality, holographic
  signage, layered light pollution, scattered atmospheric particulates.
- **Blade Runner 2049** — slow camera moves, oppressive scale, color
  blocking (orange dust → blue cold → magenta neon), silhouetted
  megastructures.

### Storyboard

| Scroll % | Camera state | Foreground | Background |
|---|---|---|---|
| 0 % | High altitude, looking forward and slightly down | **Blip** — a tethered ad-balloon with a giant LED screen cycling Bryan's messages: `:p`, `hello`, `<3`, with a glitch-pass between each (see §6). | Skyscraper tips poking through atmospheric haze |
| ~15 % | Camera begins descent, Blip drifts up + behind | Blip's tether/fans visible briefly | Skyscraper tops fully revealed; Times-Square-density signage starts to appear |
| 30–70 % | Camera continues descending past floor after floor of building screens | **Project screens** — each building face is a giant LED display showing a project (poster image / looping demo) | Adjacent buildings, parallax holograms, distant traffic streaks |
| ~80 % | Camera near street level, slows | About / contact panel embedded as a holographic kiosk or wall projection | Wet asphalt, neon reflections, foot traffic blur |
| 100 % | Camera halt at street | Footer (could be the same Blip glimpsed lifting off again — bookend) | Distant skyline, end of scroll |

### Hard requirements (Bryan's spec)

1. Cyberpunk aesthetic — no clean Apple-style minimalism.
2. Camera-down-the-city scroll metaphor.
3. Each project displayed as a building-mounted screen — every screen
   uses a video-style effect (scanlines, flicker, RGB split, refresh
   bar). Some screens host actual `<video>` textures, others host still
   images wrapped in the same effect so the look is consistent.
4. The "Blip" balloon as the opening signature element, cycling
   user-editable messages.
5. Logartis-style **experience** (scroll-narrative, not page-and-go).
6. **Rain** — both visible (particle layer + ground reflections) and
   audible (looping rain hum, occasional thunder cracks).
7. **Holographic nav** — diegetic floating glyphs replace a traditional
   nav bar; user "looks at" them to anchor-jump to sections.
8. **Easter eggs** — the Blip's screen periodically glitches into a
   self-aware one-liner.

### Soft goals (negotiable)

- **Random weather variants** — base state is rain; on some loads the
  scene initializes with light fog, heavy storm, or dry-and-clear. Adds
  rewatch value without changing layout.
- Sub-easter-eggs scattered across building screens (one of them shows
  a 404 / glitching "stay tuned" between project rotations).

### Non-goals

- Browsable map / free camera. The descent is **on-rails** —
  scroll = vertical position. No orbit controls.
- VR / AR.
- Crypto / wallet stuff (the cyberpunk aesthetic invites the trope; we
  resist).

---

## 2. Reference teardown

### logartis.info

WebFetch returned only metadata (the page is a JS-rendered SPA). Bryan
linked it as the *experience template*: scroll-driven narrative, the
content reveals itself as you descend, navigation feels diegetic. The
takeaway is **structural**, not visual: scroll is the primary input;
there's no traditional nav-as-menu.

### wodniack.dev

- **Stack:** Astro (asset hashing in `_astro/...DuGYX_YQ_Zedl3o.webp`
  signals Astro's image pipeline). Astro ships zero JS by default; only
  the animation islands hydrate.
- **Animation:** Heavy GSAP signal — the page proudly advertises "GSAP
  SOTM" wins. Probable use of **GSAP ScrollTrigger** for scroll-driven
  timelines, **GSAP timelines** for choreographed sequences, and
  smooth-scroll polish (likely Lenis or Locomotive).
- **Visual signature:** Binary-string ambient text, contrast toggle,
  large hashed image grid. Minimalist *typographic* cyberpunk-adjacent
  rather than literal cyberpunk.
- **Takeaway for us:** Astro + GSAP is a battle-tested pairing for this
  class of site. Ships less JS than our current Next.js setup.

### animejs.com

- **Stack:** Static + anime.js v4. Modular import keeps the bundle ~24
  KB.
- **Primitives we'd want:** `ScrollObserver` for scroll-linked
  animations (`autoplay: onScroll({ sync: true })` binds animation
  progress to scroll position rather than triggering on enter), Timeline
  API for sequenced choreography, `morphTo()` for SVG glyph morphing,
  `createMotionPath()` for moving things along splines (the Blip's
  drifting departure), `createDrawable()` for line-draw effects (good
  for circuit-trace ornaments).
- **Takeaway for us:** anime.js v4 is a viable lighter-weight
  alternative to GSAP for the 2D layer. Likely won't replace ScrollTrigger
  for the 3D camera bind, but pairs well for SVG/UI choreography.

---

## 3. Tech research

### Decision: how do we render the city?

The descent is fundamentally a 3D/2.5D scene. Three options, ranked by
fit:

| Option | Pros | Cons |
|---|---|---|
| **Three.js + React Three Fiber (R3F)** | First-class React integration; declarative scene graph; huge ecosystem (`drei` helpers, `postprocessing`) | Bigger initial bundle (~150 KB) than vanilla; React reconciler overhead |
| **Vanilla Three.js** | Smaller bundle; total control | More boilerplate; awkward fit with React lifecycle |
| **Pre-rendered Lottie / image sequence + parallax** | Zero 3D runtime; deterministic visuals; cheap on mobile | Loses depth / interactivity; large asset weight if high-quality |
| **2.5D parallax (CSS layers + SVG)** | Tiny payload; works everywhere | Plateau on visual fidelity — won't feel like Night City |

**Recommendation:** R3F. The scene is the product; the bundle cost is
worth it. Use `next/dynamic({ ssr: false })` so the 3D layer never
blocks first paint.

### Scroll system

| Option | Role |
|---|---|
| **GSAP ScrollTrigger** | Bind camera Y / scene state to scroll progress. Industry standard for this exact pattern. |
| **Lenis** | Smooth-scroll momentum so the camera doesn't snap. ~6 KB. |
| **anime.js ScrollObserver** | Cheaper alternative to ScrollTrigger; might work for the simpler binds (project-screen reveal). |

**Recommendation:** Lenis + GSAP ScrollTrigger for the camera. Use
anime.js sparingly for SVG/UI choreography if we like its API better.

### Asset pipeline

- **Modeling:** Blender, exported as `.glb` (glTF binary). The buildings
  can be heavily instanced — model 4–6 hero blocks, instance them with
  variation in material/scale/position.
- **Compression:**
  - `meshopt`/`Draco` for geometry (~5–10× smaller).
  - `KTX2` / Basis for textures (GPU-decodable, much smaller than PNG).
  - `gltfpack` (CLI) handles both.
- **Texture authoring:** Substance / Photoshop / Affinity. Cyberpunk
  textures are mostly grime + emissive overlays — cheap to author.
- **Video on building screens:** `THREE.VideoTexture` mapped onto
  building-face planes. Mute, loop, autoplay, `playsInline`. Each
  screen is one project; the project's existing card data (`title`,
  `media`, `description`, `links`) survives — only the *delivery*
  changes.

### Cyberpunk look — concrete techniques

- **Bloom + tone mapping** — `postprocessing` library (`UnrealBloomPass`,
  ACES tone mapping). Drives the wet-neon glow.
- **Volumetric fog** — `THREE.FogExp2` baseline; for a richer look,
  raymarched fog shader or Drei's `<Cloud>`.
- **Color palette:** anchor on three colors max. Suggestion:
  - Magenta neon `#ff2bd6`
  - Cyan accent `#00f6ff` (already in the existing palette)
  - Dirty amber `#ffae42` for warm interiors
  - Deep blue-black `#070914` ambient
- **Atmospherics:**
  - Light-pollution gradient sky shader.
  - **Rain:** `InstancedMesh` of thin angled streaks, additive blend,
    instance count scales with weather intensity. Wet-asphalt reflection
    plane at street level (planar reflection or matcap fake).
  - **Lightning + thunder:** every 8–25 s under "storm" weather, flash
    a high-intensity directional light for ~80 ms then fade. Pair with
    a thunder sample, audio-only (no shake under reduce-motion).
  - **Weather variants** — pick one at session init from
    `["light_rain", "heavy_storm", "fog_drift", "dry_clear"]`. Optional
    URL override (`?weather=storm`) for sharing a specific look.
  - Volumetric god-rays from screens (cheap fake: light-cone meshes).
- **Glitch + scan lines on screens** — every screen — video and still
  alike — gets the same shader pass: scanlines, RGB split, a vertical
  refresh bar, and per-frame jitter at 1–2 px. Stills become "video"
  via this overlay; real videos get the same treatment for visual unity.
- **Camera shake** — subtle 3-axis sine-noise on the camera transform
  to suggest atmosphere/wind. Off under reduce-motion.

### Typography

The current Manrope reads modern but neutral. For cyberpunk, swap or
augment to:

- **Display:** *Neue Machina* (paid, gold standard) — or free
  alternatives: *Orbitron*, *Chakra Petch*, *Rajdhani*.
- **Body:** Keep Manrope (it's clean and reads well at small sizes).
- **Accent / glitch text:** Variable monospace like *JetBrains Mono*
  with stroke-width animation, or *Tomorrow* for retro-techno feel.

### Performance budget

The 3D scene must not regress what the perf branch just shipped. Hard
budget:

- **Initial JS:** ≤ 120 KB gzipped (excluding the lazy 3D bundle).
- **3D bundle (lazy):** ≤ 350 KB gzipped (R3F + drei + postprocessing
  is ~250 KB; leave headroom).
- **glb assets total:** ≤ 4 MB after gltfpack.
- **Video screens:** each ≤ 600 KB H.264; max 4 playing simultaneously
  (cull off-screen).
- **Mobile:** below 600 px width, swap the 3D scene for a tall scrollable
  poster (parallax SVG composition). Consider `prefers-reduced-data`
  also degrading to the poster.

### Accessibility

Non-negotiable, even with the spectacle:

- `prefers-reduced-motion` → poster fallback or static "still" of the
  scene.
- All project info reachable as plain HTML below or inside a "skip
  cinematic" link at the top.
- Building screens' content (project title, blurb, link) must be in
  the DOM as text, not just baked into the texture.
- Color contrast on captions ≥ WCAG AA. Magenta-on-deep-blue passes;
  cyan-on-deep-blue is borderline — verify per element.

---

## 4. Stack decisions

### Stack: migrating to Astro

| | Next.js (current) | Astro (target) |
|---|---|---|
| Bundle baseline | ~80 KB gzipped framework | ~0 KB by default; islands hydrate on demand |
| 3D fit | R3F is first-class on Next | Works fine as a React island; the rest of the page ships zero JS |
| Migration cost | Zero | ~1–2 days to port pages, content, image config |
| Long-term ergonomics | Familiar, integrated | Less JS shipped per page, simpler mental model, multi-framework islands |

**Decision: migrate to Astro.** Bryan asked for "whatever is best in
the long run." Earlier I recommended staying on Next because the
migration cost wasn't justified by perf alone. That argument loses
weight here: the refactor is total — the pages, hero, and footer are
all being rewritten — so there's not much to "migrate" except routing
and content. The whole site outside the 3D island then ships near-zero
JS, which matches the wodniack benchmark we're chasing. The 3D scene
becomes a single React island lazy-loaded on the home route.

Pick Astro's React integration (`@astrojs/react`) so we can keep R3F.
Use Astro's image pipeline (`astro:assets`) for the project posters —
it produces hashed AVIF/WebP variants without the Next-image runtime.
API routes (currently `apps/web/app/api/chat/route.ts` for the chatbot)
become Astro endpoints; behavior is identical.

### Add or replace?

**Add:** `astro`, `@astrojs/react`, `three`, `@react-three/fiber`,
`@react-three/drei`, `postprocessing`, `lenis`, `gsap`, `howler`
(audio).

**Maybe add:** `animejs` (v4) — for SVG/UI flourishes (hologram nav
glyph morphs, scanline overlays) if GSAP feels heavy for those.

**Drop:** the entire Next.js scaffolding once Astro is wired up —
`apps/web/app/`, `next.config.mjs`, `next-env.d.ts`,
`@bamp/web` package's Next deps. Keep the existing components on a
reference branch (`archive/black-hole-hero`) until the new scene is
shippable, then delete.

---

## 5. Implementation phases

The refactor is large. Phasing it lets us ship something visible early
and protects against the trap of a six-month-rebuild that never lands.

### Phase 0 — pre-work

- Create `refactor/cyberpunk-city` branch from main (after perf PR
  merges).
- Stand up Astro skeleton alongside the existing Next.js app, then
  cut over routes once parity is reached.
- Lock the color palette + collect mood-board imagery in
  `docs/brand/cyberpunk/`.
- Pick fonts; install via Astro's font integration (or self-host if
  Neue Machina).
- Source/license rain + thunder audio samples, store in
  `public/audio/`.

### Phase 1 — scene scaffold (3–5 days)

- Wire R3F into the route with a placeholder cube + camera scroll bind.
- Add Lenis + GSAP ScrollTrigger plumbing.
- Establish the lighting + bloom + tone mapping baseline.
- Camera path: y from `+altitudeMax` → `+streetLevel` linked to scroll.

### Phase 2 — Blip + skybox (3–4 days)

- Model the Blip in Blender (~half day for low-poly).
- Animate its drift, screen flicker, slow rotation.
- Skybox: light-pollution gradient shader; distant skyline silhouette
  card.
- Atmospheric haze.

### Phase 3 — building grid + screens (5–7 days)

- Procedural placement of building blocks along the descent (instanced
  mesh).
- Building-face plane geometry parameterized to host video/image
  textures.
- Project data → screen texture pipeline. Reuse existing
  `content/projects.ts`.
- Lazy-load video textures; cull off-screen.

### Phase 4 — atmospherics + audio + polish

- Rain particle system + wet-asphalt reflection plane.
- Lightning flash + thunder audio. Randomized timing under storm
  weather.
- Weather variant picker (random at session init, optional URL flag).
- God-rays / light cones from screens.
- Camera shake.
- Glitch shader pass tied to scroll speed (faster scroll → more glitch).
- Audio system (Howler) — rain bed, thunder, optional pad, mute toggle,
  `localStorage` persistence.
- Holographic nav glyphs — diegetic anchor links, glitch on hover.
- Easter-egg copy file (`src/content/blip.ts`), Blip message rotator.

### Phase 5 — fallbacks + a11y (2–3 days)

- Mobile poster (tall SVG/CSS layered parallax).
- `prefers-reduced-motion` → static cinematic still + plain project
  cards below.
- Keyboard nav: tab order across project screens via DOM mirror.

### Phase 6 — content + ship (2 days)

- Re-evaluate which project screens get video vs poster.
- Final color/lighting passes.
- Lighthouse + bundle audit.
- Merge.

No deadline (Bryan's call): build at quality, not at speed. Phases are
sequenced for *risk reduction* — phase 1 proves the camera bind works
before any modeling investment; phase 3 proves the screen-texture
pipeline before atmospherics get layered on top.

---

## 6. Decisions (locked)

Confirmed with Bryan 2026-04-28.

1. **Cyberpunk reference set:** Cyberpunk: Edgerunners, Cyberpunk 2077,
   **Blade Runner 2049**.
2. **Blip messages:** rotating, user-editable list. Initial set:
   `:p`, `hello`, `<3`. Glitch transition between each. The list lives
   in a content file (`src/content/blip.ts` post-migration) so Bryan
   can append without touching components.
3. **Project screens:** every screen uses a video-style shader (see
   §3). Most screens texture-map a still image; one screen hosts the
   actual `link-app-demo.mp4` (already `preload="metadata"` via the
   perf branch — re-encode still pending).
4. **Audio:** in scope.
   - Looping rain bed (default on, with a clearly-visible mute toggle).
   - Thunder samples, randomized timing under storm weather.
   - Optional ambient synth pad layered low.
   - Default volume: low. Mute under `prefers-reduced-motion`. Persist
     mute choice in `localStorage`.
5. **Mobile:** different experience is fine. Tall scroll-poster with
   parallax SVG layers, no 3D scene, no audio autoload.
6. **Stack:** **Astro** (see §4). Bryan deferred to my call; with the
   refactor being total, the migration cost is absorbed.
7. **Timeline:** none. Build at quality, not at deadline.

---

## 7. What we'll write next

- `docs/REFACTOR_PLAN.md` — concrete task list per phase with file
  paths and acceptance criteria.
- `docs/brand/cyberpunk/` — palette, fonts, mood imagery, screen-frame
  templates.
- `docs/brand/scene-camera.md` — camera path keyframes, FOV, fog
  parameters, lighting rig.
- `docs/brand/audio.md` — sample list, licensing, default volumes,
  weather-to-audio mapping.

---

## Appendix — library shortlist

| Package | Purpose | Approx. gzipped |
|---|---|---|
| `astro` + `@astrojs/react` | Site framework + React island runtime | host-side; islands ship per-component |
| `three` | Core 3D | 130 KB |
| `@react-three/fiber` | React renderer for Three | 30 KB |
| `@react-three/drei` | Helpers (cameras, loaders, postprocessing wrappers) | varies — tree-shake |
| `postprocessing` | Bloom, tone mapping, glitch passes | 30 KB |
| `lenis` | Smooth scroll | 6 KB |
| `gsap` + `ScrollTrigger` | Scroll-driven timelines | 50 KB combined |
| `howler` | Audio (rain, thunder, ambient pad) | 10 KB |
| `animejs` (v4, optional) | SVG/UI choreography for hologram nav | 24 KB |
