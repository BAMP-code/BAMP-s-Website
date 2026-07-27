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
  slug, title, description, status, category_id,
  media_type, media_src, media_alt, media_width, media_height, sort_order
)
select
  v.slug, v.title, v.description, v.status, c.id,
  v.media_type, v.media_src, v.media_alt, v.media_width, v.media_height, v.sort_order
from (
  values
    ('waypoint-prediction', 'End-To-End Waypoint Prediction',
     'A deep learning project for predicting waypoints in autonomous navigation using the nuScenes dataset.',
     'completed', 'cs', 'image', '/images/waypoint_prediction.jpg', 'End-To-End Waypoint Prediction', 700, 400, 0),

    ('bio-printing-ui', 'Bio Printing User Interface',
     'Developing a friendly user interface where users can view and interact with 3D documents of human vessels.',
     'completed', 'cs', 'image', '/images/GUI.jpg', 'Bio Printing User Interface', 700, 400, 1),

    ('website-for-her', 'Website for Her',
     'A website created for the sole purpose of asking out my girlfriend, Daniela, on a date.',
     'completed', 'cs', 'image', '/images/for-her.jpg', 'Website for Her', 700, 400, 2),

    ('link-app', 'L''Ink App',
     'A mobile app for collaborative note-taking, sketching, and sharing.',
     'in-progress', 'cs', 'video', '/videos/link-app-demo.mp4', 'L''Ink App demo', 390, 844, 3),

    ('ecg', 'Electro Cardiogram',
     'A project focused on building and analyzing an ECG circuit for biomedical applications.',
     'completed', 'ee-me', 'image', '/images/ECG_project.jpg', 'Electro Cardiogram Project', 700, 400, 4),

    ('truss', 'Truss',
     'Mechanical engineering project involving the design and analysis of a truss structure.',
     'completed', 'ee-me', 'image', '/images/Truss_project.jpg', 'Truss Project', 700, 400, 5),

    ('led-board', 'LED Board',
     'Designed and built a custom LED board for interactive displays, we are able to play ping pong on it.',
     'completed', 'ee-me', 'image', '/images/LED_board.jpg', 'LED Board Project', 700, 400, 6),

    ('useless-box', 'Useless Box',
     'A fun electronics project: a box that turns itself off when you turn it on! It has other modes like a shy box.',
     'completed', 'ee-me', 'image', '/images/Useless_box.jpg', 'Useless Box Project', 700, 400, 7),

    ('cook-drawing', '"Cook"',
     'The drawing is that of a friend''s dog, his name is Coco, but we call him Coook. This was my first time taking on a serious drawing project.',
     'completed', 'drawings', 'image', '/images/Cook.jpg', 'Cook drawing', 700, 400, 8),

    ('unnamed-drawing', '"Unnamed"',
     'The drawing was inspired by my research paper about patients suffering from schizophrenia.',
     'completed', 'drawings', 'image', '/images/Ghost.jpg', 'Unnamed drawing', 700, 400, 9)
) as v(slug, title, description, status, category_slug, media_type, media_src, media_alt, media_width, media_height, sort_order)
join categories c on c.slug = v.category_slug
on conflict (slug) do nothing;
