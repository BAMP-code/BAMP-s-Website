import { z } from "zod";
import { supabase } from "./supabase";

// Project/category data lives in Supabase (see supabase/schema.sql) —
// nothing about categories or counts is hardcoded here. Rows are
// Zod-validated on the way in, so a malformed row fails the build loudly
// instead of silently rendering broken content.

const categorySchema = z.object({
  id: z.string(),
  slug: z.string(),
  label: z.string(),
  sort_order: z.number(),
});

const projectRowSchema = z.object({
  id: z.string(),
  slug: z.string(),
  title: z.string(),
  description: z.string(),
  status: z.enum(["completed", "in-progress"]),
  media_type: z.enum(["image", "video"]),
  media_src: z.string(),
  media_alt: z.string(),
  media_width: z.number(),
  media_height: z.number(),
  sort_order: z.number(),
  categories: categorySchema,
});

export type Category = z.infer<typeof categorySchema>;

export type Project = {
  id: string;
  slug: string;
  title: string;
  description: string;
  status: "completed" | "in-progress";
  category: Category;
  media: {
    type: "image" | "video";
    src: string;
    alt: string;
    width: number;
    height: number;
  };
};

// Astro's build evaluates each module once per process, so this module-
// level cache is shared across every .astro file that calls getProjects()
// / getCategories() during the same `astro build` — one round trip total.
let projectsPromise: Promise<Project[]> | null = null;
let categoriesPromise: Promise<Category[]> | null = null;

async function fetchCategories(): Promise<Category[]> {
  const { data, error } = await supabase
    .from("categories")
    .select("*")
    .order("sort_order", { ascending: true });

  if (error) {
    throw new Error(`Failed to fetch categories from Supabase: ${error.message}`);
  }
  return z.array(categorySchema).parse(data);
}

async function fetchProjects(): Promise<Project[]> {
  const { data, error } = await supabase
    .from("projects")
    .select("*, categories(*)")
    .order("sort_order", { ascending: true });

  if (error) {
    throw new Error(`Failed to fetch projects from Supabase: ${error.message}`);
  }

  return z
    .array(projectRowSchema)
    .parse(data)
    .map((row) => ({
      id: row.id,
      slug: row.slug,
      title: row.title,
      description: row.description,
      status: row.status,
      category: row.categories,
      media: {
        type: row.media_type,
        src: row.media_src,
        alt: row.media_alt,
        width: row.media_width,
        height: row.media_height,
      },
    }));
}

export function getCategories(): Promise<Category[]> {
  if (!categoriesPromise) categoriesPromise = fetchCategories();
  return categoriesPromise;
}

export function getProjects(): Promise<Project[]> {
  if (!projectsPromise) projectsPromise = fetchProjects();
  return projectsPromise;
}
