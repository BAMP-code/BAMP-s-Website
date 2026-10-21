-- bamp.codes — project content schema + seed data.
--
-- Run this once in the Supabase SQL editor (Project → SQL Editor → New
-- query → paste → Run). It's idempotent: re-running it won't duplicate
-- rows or error on tables that already exist.
--
-- After this: add/edit/reorder projects and categories directly in the
-- Supabase Table Editor. The site reads this data at build time (see
-- apps/web/src/lib/projects.ts) — a Vercel redeploy (git push, or
-- `vercel --prod`) picks up any changes.

create table if not exists categories (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  label text not null,
  sort_order integer not null default 0
);

create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  title text not null,
  description text not null,
  case_body text not null,
  case_close text not null,
  tech_tags text[] not null default '{}',
  status text not null check (status in ('completed', 'in-progress')),
  category_id uuid not null references categories(id) on delete restrict,
  media_type text not null check (media_type in ('image', 'video')),
  media_src text not null,
  media_alt text not null,
  media_width integer not null,
  media_height integer not null,
  sort_order integer not null default 0,
  created_at timestamptz not null default now()
);

alter table categories enable row level security;
alter table projects enable row level security;

drop policy if exists "Public read access" on categories;
create policy "Public read access" on categories for select using (true);

drop policy if exists "Public read access" on projects;
create policy "Public read access" on projects for select using (true);

-- No insert/update/delete policies are defined, so the public anon key
-- used by the site build can only ever read this data — writes only
-- happen through the Supabase dashboard (or the SQL editor), which
-- connects with elevated privileges and bypasses RLS.

insert into categories (slug, label, sort_order) values
  ('cs', 'CS', 0),
  ('ee-me', 'EE·ME', 1),
  ('drawings', 'DRAW', 2)
on conflict (slug) do nothing;

insert into projects (
  slug, title, description, case_body, case_close, tech_tags, status, category_id,
  media_type, media_src, media_alt, media_width, media_height, sort_order
)
select
  v.slug, v.title, v.description, v.case_body, v.case_close, v.tech_tags, v.status, c.id,
  v.media_type, v.media_src, v.media_alt, v.media_width, v.media_height, v.sort_order
from (
  values
    ('waypoint-prediction', 'End-To-End Waypoint Prediction',
     'A deep learning project for predicting waypoints in autonomous navigation using the nuScenes dataset.',
     'Self-driving cars have to guess where everything around them is about to go, fast, with partial information. I wanted to actually build a model that does that, not just read papers about it. I trained a deep learning pipeline on nuScenes, real autonomous-driving sensor data, going from raw inputs straight to predicted waypoints; most of the time went into getting the loss function to reflect what "good" actually looks like, not just low error.',
     'Ended up with a model that holds up against the nuScenes benchmarks, and a much better gut sense for why self-driving is such a hard problem.',
     array['Python', 'PyTorch', 'nuScenes', 'Deep Learning'],
     'completed', 'cs', 'image', '/images/waypoint_prediction.jpg', 'End-To-End Waypoint Prediction', 700, 400, 0),

    ('bio-printing-ui', 'Bio Printing User Interface',
     'Developing a friendly user interface where users can view and interact with 3D documents of human vessels.',
     'A bioprinting lab had 3D scans of vascular structures that only made sense as raw volumetric data. Nobody without a CS background could actually look at them. So I built a GUI that lets researchers rotate, slice, and inspect 3D vessel models directly, instead of parsing point clouds by hand.',
     'Researchers started using it day-to-day, which, for a student side project, was the whole point.',
     array['C/C++', '3D Visualization', 'Biomedical', 'GUI'],
     'completed', 'cs', 'image', '/images/GUI.jpg', 'Bio Printing User Interface', 700, 400, 1),

    ('website-for-her', 'Website for Her',
     'A website created for the sole purpose of asking out my girlfriend, Daniela, on a date.',
     'I wanted to ask my girlfriend Daniela out, and a text felt like it undersold the effort, so I built a small site from scratch just for her: its own copy, layout, and probably more attention to detail than something only one person would ever see really needed.',
     'She said yes. Still the best-received launch I''ve shipped.',
     array['HTML/CSS', 'JavaScript', 'Personal Project'],
     'completed', 'cs', 'image', '/images/for-her.jpg', 'Website for Her', 700, 400, 2),

    ('link-app', 'L''Ink App',
     'A mobile app for collaborative note-taking, sketching, and sharing.',
     'Most note-taking apps are built for typing, not drawing, and the ones that support sketching rarely feel good to share or collaborate in. I''m building a mobile app for collaborative sketching and note-taking, with a real-time sync layer so two people can draw on the same canvas without it feeling laggy.',
     'Still in progress. Current builds already support live shared canvases; UI polish is next.',
     array['Mobile', 'React Native', 'Real-time Sync'],
     'in-progress', 'cs', 'video', '/videos/link-app-demo.mp4', 'L''Ink App demo', 390, 844, 3),

    ('ecg', 'Electro Cardiogram',
     'A project focused on building and analyzing an ECG circuit for biomedical applications.',
     'I wanted to actually understand how an ECG works at the circuit level, not just know the textbook diagram, so I designed and built the amplification and filtering stages myself. Spent a while figuring out why my "clean" signal kept picking up 60Hz noise from everything else in the room.',
     'Ended up with a circuit that reliably reads a real heartbeat, and a much better intuition for analog signal chains.',
     array['Circuit Design', 'Biomedical', 'Signal Processing'],
     'completed', 'ee-me', 'image', '/images/ECG_project.jpg', 'Electro Cardiogram Project', 700, 400, 4),

    ('truss', 'Truss',
     'Mechanical engineering project involving the design and analysis of a truss structure.',
     'Wanted hands-on proof that the structural analysis math I''d been doing on paper actually holds up. I designed a truss structure and iterated on the geometry, pushing for the best strength-to-material ratio I could justify with the load calculations.',
     'A design that held up to the numbers, and a better feel for why engineers obsess over triangles.',
     array['Mechanical Design', 'Structural Analysis', 'CAD'],
     'completed', 'ee-me', 'image', '/images/Truss_project.jpg', 'Truss Project', 700, 400, 5),

    ('led-board', 'LED Board',
     'Designed and built a custom LED board for interactive displays, we are able to play ping pong on it.',
     'Wanted to build something physical and interactive from scratch, not just the display, but the electronics and the game logic driving it too. I designed and built a custom LED board, wiring the hardware and writing the control logic myself.',
     'It plays Pong. Genuinely one of the most fun things I''ve built.',
     array['Embedded Systems', 'Hardware', 'Electronics'],
     'completed', 'ee-me', 'image', '/images/LED_board.jpg', 'LED Board Project', 700, 400, 6),

    ('useless-box', 'Useless Box',
     'A fun electronics project: a box that turns itself off when you turn it on! It has other modes like a shy box.',
     'The classic beginner electronics build: a box whose entire function is turning itself off. I built the mechanism and control logic, then kept going past the obvious version and added a few extra personality modes, including one that acts "shy."',
     'A fully working useless machine that''s somehow more entertaining than most of my useful projects.',
     array['Electronics', 'Mechanical Design', 'Microcontroller'],
     'completed', 'ee-me', 'image', '/images/Useless_box.jpg', 'Useless Box Project', 700, 400, 7),

    ('cook-drawing', '"Cook"',
     'The drawing is that of a friend''s dog, his name is Coco, but we call him Coook. This was my first time taking on a serious drawing project.',
     'Wanted to draw my friend''s dog, Coco (we call him Cook), properly, instead of the quick sketches I was used to. My first real attempt at a realistic portrait, and I spent most of the time on fur texture, getting the lighting to actually read as fur and not fuzz.',
     'First finished portrait-style piece I was genuinely proud of.',
     array['Graphite', 'Portrait', 'Traditional Art'],
     'completed', 'drawings', 'image', '/images/Cook.jpg', 'Cook drawing', 700, 400, 8),

    ('unnamed-drawing', '"Unnamed"',
     'The drawing was inspired by my research paper about patients suffering from schizophrenia.',
     'I''d just read a research paper on schizophrenia and had a lot I wanted to process that wasn''t going to come out as a sentence, so I drew it instead, leaning into fragmentation and distorted perspective to get at something the paper''s language couldn''t.',
     'The piece that convinced me drawing and engineering aren''t that different; both are just ways of making an idea visible.',
     array['Illustration', 'Conceptual', 'Traditional Art'],
     'completed', 'drawings', 'image', '/images/Ghost.jpg', 'Unnamed drawing', 700, 400, 9)
) as v(slug, title, description, case_body, case_close, tech_tags, status, category_slug, media_type, media_src, media_alt, media_width, media_height, sort_order)
join categories c on c.slug = v.category_slug
on conflict (slug) do nothing;
