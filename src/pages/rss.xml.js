import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { getPublishedPosts } from "../utils/blog";

export async function GET(context) {
  const posts = getPublishedPosts(await getCollection("blog"));

  return rss({
    title: "Hangwoo Cho | Engineering Notes",
    description: "Notes on AI / ML engineering and the lessons learned along the way.",
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.publishDate,
      link: `/blog/${post.slug}/`
    }))
  });
}