import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { getPublishedPosts } from "../utils/blog";

export async function GET(context) {
  const posts = getPublishedPosts(await getCollection("blog"));

  return rss({
    title: "Hangwoo Cho | Engineering Notes",
    description: "회사에서 진행한 ML과 LLM 프로젝트 경험을 기록하는 개발 블로그",
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.publishDate,
      link: `/blog/${post.slug}/`
    }))
  });
}