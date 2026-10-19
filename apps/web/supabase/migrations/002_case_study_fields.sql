-- Adds case-study copy (body + close paragraphs) and tech-stack tags to
-- the projects table, replacing the single `description` field in the
-- work-list UI. `description` is left in place, just unused by the UI —
-- this migration only adds columns, nothing is dropped.
--
-- Run this once in the Supabase SQL editor against an existing project
-- (one that already ran schema.sql). A fresh project seeded from the
-- updated schema.sql already has these columns and values — skip this
-- file in that case. Idempotent: safe to re-run.

alter table projects add column if not exists case_body text;
alter table projects add column if not exists case_close text;
alter table projects add column if not exists tech_tags text[] not null default '{}';

update projects set
  case_body = v.case_body,
  case_close = v.case_close,
  tech_tags = v.tech_tags
from (
  values
    ('waypoint-prediction',
     'Self-driving cars have to guess where everything around them is about to go, fast, with partial information. I wanted to actually build a model that does that, not just read papers about it. I trained a deep learning pipeline on nuScenes, real autonomous-driving sensor data, going from raw inputs straight to predicted waypoints; most of the time went into getting the loss function to reflect what "good" actually looks like, not just low error.',
     'Ended up with a model that holds up against the nuScenes benchmarks, and a much better gut sense for why self-driving is such a hard problem.',
     array['Python', 'PyTorch', 'nuScenes', 'Deep Learning']),

    ('bio-printing-ui',
     'A bioprinting lab had 3D scans of vascular structures that only made sense as raw volumetric data. Nobody without a CS background could actually look at them. So I built a GUI that lets researchers rotate, slice, and inspect 3D vessel models directly, instead of parsing point clouds by hand.',
     'Researchers started using it day-to-day, which, for a student side project, was the whole point.',
     array['C/C++', '3D Visualization', 'Biomedical', 'GUI']),

    ('website-for-her',
     'I wanted to ask my girlfriend Daniela out, and a text felt like it undersold the effort, so I built a small site from scratch just for her: its own copy, layout, and probably more attention to detail than something only one person would ever see really needed.',
     'She said yes. Still the best-received launch I''ve shipped.',
     array['HTML/CSS', 'JavaScript', 'Personal Project']),

    ('link-app',
     'Most note-taking apps are built for typing, not drawing, and the ones that support sketching rarely feel good to share or collaborate in. I''m building a mobile app for collaborative sketching and note-taking, with a real-time sync layer so two people can draw on the same canvas without it feeling laggy.',
     'Still in progress. Current builds already support live shared canvases; UI polish is next.',
     array['Mobile', 'React Native', 'Real-time Sync']),

    ('ecg',
     'I wanted to actually understand how an ECG works at the circuit level, not just know the textbook diagram, so I designed and built the amplification and filtering stages myself. Spent a while figuring out why my "clean" signal kept picking up 60Hz noise from everything else in the room.',
     'Ended up with a circuit that reliably reads a real heartbeat, and a much better intuition for analog signal chains.',
     array['Circuit Design', 'Biomedical', 'Signal Processing']),

    ('truss',
     'Wanted hands-on proof that the structural analysis math I''d been doing on paper actually holds up. I designed a truss structure and iterated on the geometry, pushing for the best strength-to-material ratio I could justify with the load calculations.',
     'A design that held up to the numbers, and a better feel for why engineers obsess over triangles.',
     array['Mechanical Design', 'Structural Analysis', 'CAD']),

    ('led-board',
     'Wanted to build something physical and interactive from scratch, not just the display, but the electronics and the game logic driving it too. I designed and built a custom LED board, wiring the hardware and writing the control logic myself.',
     'It plays Pong. Genuinely one of the most fun things I''ve built.',
     array['Embedded Systems', 'Hardware', 'Electronics']),

    ('useless-box',
     'The classic beginner electronics build: a box whose entire function is turning itself off. I built the mechanism and control logic, then kept going past the obvious version and added a few extra personality modes, including one that acts "shy."',
     'A fully working useless machine that''s somehow more entertaining than most of my useful projects.',
     array['Electronics', 'Mechanical Design', 'Microcontroller']),

    ('cook-drawing',
     'Wanted to draw my friend''s dog, Coco (we call him Cook), properly, instead of the quick sketches I was used to. My first real attempt at a realistic portrait, and I spent most of the time on fur texture, getting the lighting to actually read as fur and not fuzz.',
     'First finished portrait-style piece I was genuinely proud of.',
     array['Graphite', 'Portrait', 'Traditional Art']),

    ('unnamed-drawing',
     'I''d just read a research paper on schizophrenia and had a lot I wanted to process that wasn''t going to come out as a sentence, so I drew it instead, leaning into fragmentation and distorted perspective to get at something the paper''s language couldn''t.',
     'The piece that convinced me drawing and engineering aren''t that different; both are just ways of making an idea visible.',
     array['Illustration', 'Conceptual', 'Traditional Art'])
) as v(slug, case_body, case_close, tech_tags)
where projects.slug = v.slug;

alter table projects alter column case_body set not null;
alter table projects alter column case_close set not null;
