---
layout: default
title: Articles
---
<h1>Articles</h1>
<ul class="article-list">
  {% for post in site.posts %}
  <li class="article-item">
    <span class="article-date">{{ post.date | date: "%B %d, %Y" }}</span>
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    {% if post.excerpt %}
    <p class="article-excerpt">{{ post.excerpt }}</p>
    {% endif %}
  </li>
  {% endfor %}
</ul>
