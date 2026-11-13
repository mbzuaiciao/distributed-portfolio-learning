import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';

export interface TocItem {
  id: string;
  text: string;
  depth: number;
}

export interface MarkdownRenderResult {
  html: string;
  toc: TocItem[];
}

/**
 * Remark plugin to transform GitHub-style callouts:
 * > [!NOTE]
 * > Content...
 */
function remarkCallouts() {
  return (tree: any) => {
    visit(tree, 'blockquote', (node: any) => {
      const firstChild = node.children?.[0];
      if (firstChild?.type === 'paragraph' && firstChild.children?.[0]?.type === 'text') {
        const textNode = firstChild.children[0];
        const match = textNode.value.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\r?\n)?/i);
        if (match) {
          const type = match[1].toUpperCase();
          textNode.value = textNode.value.slice(match[0].length);
          // If the text node is now empty, clean it up
          if (textNode.value.trim() === '' && firstChild.children.length > 1) {
            firstChild.children.shift();
          }
          node.data = node.data || {};
          node.data.hName = 'aside';
          node.data.hProperties = {
            className: ['callout', `callout-${type.toLowerCase()}`],
            'data-callout-type': type.toLowerCase(),
          };
          node.children.unshift({
            type: 'paragraph',
            data: {
              hName: 'div',
              hProperties: { className: ['callout-title'] },
            },
            children: [{ type: 'text', value: type }],
          });
        }
      }
    });
  };
}

/**
 * Rehype plugin to rewrite relative markdown links:
 * - 01_why_distributed.md -> /learn/01_why_distributed
 * - ../docs/foo.md -> /experiments/docs/foo
 * - ../artifacts/bar/report.md -> /experiments/artifacts/bar/report
 */
function rehypeRewriteLinks() {
  return (tree: any) => {
    visit(tree, 'element', (node: any) => {
      if (node.tagName === 'a' && node.properties?.href) {
        const href = String(node.properties.href);

        // Relative tutorial link: "01_why_distributed.md" or "./01_why_distributed.md"
        const tutorialMatch = href.match(/^(?:\.\/)?(\d{2}_[a-zA-Z0-9_-]+)\.md(#.*)?$/);
        if (tutorialMatch) {
          node.properties.href = `/learn/${tutorialMatch[1]}${tutorialMatch[2] || ''}`;
          return;
        }

        // Relative docs link: "../docs/something.md" or "docs/something.md"
        const docMatch = href.match(/^(?:\.\.\/)*docs\/([a-zA-Z0-9_/-]+)\.md(#.*)?$/);
        if (docMatch) {
          node.properties.href = `/experiments/docs/${docMatch[1]}${docMatch[2] || ''}`;
          return;
        }

        // Relative artifacts link: "../artifacts/something.md" or "artifacts/something.md"
        const artifactMatch = href.match(/^(?:\.\.\/)*artifacts\/([a-zA-Z0-9_/-]+)\.md(#.*)?$/);
        if (artifactMatch) {
          node.properties.href = `/experiments/artifacts/${artifactMatch[1]}${artifactMatch[2] || ''}`;
          return;
        }
      }
    });
  };
}

/**
 * Rehype plugin to collect headings for the in-page Table of Contents
 */
function rehypeExtractToc(tocList: TocItem[]) {
  return (tree: any) => {
    visit(tree, 'element', (node: any) => {
      if ((node.tagName === 'h2' || node.tagName === 'h3') && node.properties?.id) {
        const depth = parseInt(node.tagName.slice(1), 10);
        let text = '';
        visit(node, 'text', (t: any) => {
          text += t.value;
        });
        if (text.trim()) {
          tocList.push({
            id: String(node.properties.id),
            text: text.trim(),
            depth,
          });
        }
      }
    });
  };
}

/**
 * Render Markdown content to HTML with full LaTeX and callout support
 */
export async function renderMarkdown(content: string): Promise<MarkdownRenderResult> {
  const toc: TocItem[] = [];

  const processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkCallouts)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeKatex)
    .use(rehypeSlug)
    .use(rehypeExtractToc, toc)
    .use(rehypeRewriteLinks)
    .use(rehypeStringify, { allowDangerousHtml: true });

  const file = await processor.process(content);

  return {
    html: String(file),
    toc,
  };
}
