# AGENTS.md

This is a Jekyll-based GitHub Pages blog. Below are the key conventions and instructions for working with this repo.

## Project Structure

```
├── _config.yml          # Jekyll site configuration
├── _layouts/            # HTML layouts (article.html, default.html)
├── _posts/              # Blog posts (YYYY-MM-DD-title.md)
├── assets/              # Static files (images, CSS)
│   ├── css/style.css
│   ├── images/<post-date>-<post-slug>/  # Images per post
│   └── logo.jpg
├── index.md             # Home page
├── about.md             # About page
├── projects.md          # Projects page
└── publications.md      # Publications page
```

## Blog Post Conventions

### File Naming

- Format: `YYYY-MM-DD-slug.md`
- Example: `2026-04-03-flashattention-analysis.md`
- Place in `_posts/` directory

### Front Matter

Every post must start with YAML front matter:

```yaml
---
layout: article
title: "Post Title"
date: YYYY-MM-DD HH:MM:SS ±ZZZZ
---
```

### Referencing Images

- Store images in `assets/images/YYYY-MM-DD-slug/`
- Reference in markdown: `![alt](/assets/images/YYYY-MM-DD-slug/filename.png)`

## Editing Guidelines

1. **New posts**: Create `_posts/YYYY-MM-DD-slug.md` with proper front matter
2. **Images**: Create matching directory under `assets/images/` for each post
3. **Existing posts**: Edit in place, preserve front matter and formatting
4. **No emojis** unless explicitly requested
5. **Keep code style consistent** with existing posts

## Build & Preview

- This is a GitHub Pages site; no local build step required
- Changes are deployed automatically on push to `main`

## Git Workflow

- Commit messages should be concise and descriptive
- Do not commit unless explicitly asked
