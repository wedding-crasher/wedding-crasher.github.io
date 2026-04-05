import type { CollectionEntry } from "astro:content";

export type BlogEntry = CollectionEntry<"blog">;

export function slugifyCategory(value: string) {
  return value
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function getPublishedPosts(posts: BlogEntry[]) {
  return posts
    .filter((post) => !post.data.draft)
    .sort((left, right) => right.data.publishDate.getTime() - left.data.publishDate.getTime());
}

export function getFeaturedPost(posts: BlogEntry[]) {
  return posts.find((post) => post.data.featured) ?? posts[0];
}

export function getCategories(posts: BlogEntry[]) {
  const counts = new Map<string, number>();

  for (const post of posts) {
    const currentCount = counts.get(post.data.category) ?? 0;
    counts.set(post.data.category, currentCount + 1);
  }

  return [...counts.entries()]
    .map(([label, count]) => ({
      label,
      count,
      slug: slugifyCategory(label)
    }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

export function formatDate(value: Date) {
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "long",
    day: "numeric"
  }).format(value);
}