const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const page = await browser.newPage({ url: "http://127.0.0.1:8099/" });
await sleep(2500);
const ev = (fn) => page.evaluate(fn);
const out = {};
out.title = await ev(() => document.title);
out.insBefore = await ev(() => document.getElementById("ins-body").textContent.slice(0, 80));
out.mainClassBefore = await ev(() => document.getElementById("main").className);
await page.click("#btn-rec");
await sleep(4000);
out.insAfter = await ev(() => document.getElementById("ins-body").textContent.slice(0, 150));
out.insTitleAfter = await ev(() => document.getElementById("ins-title").textContent);
out.mainClassAfter = await ev(() => document.getElementById("main").className);
out.diagramNodes = await ev(() => document.querySelectorAll("#diagram .node").length);
out.clicked = await ev(() => {
  const n = document.querySelector('#diagram .node[data-id="b0.deltanet"]');
  if (n) { n.dispatchEvent(new MouseEvent("click", { bubbles: true })); return true; }
  return false;
});
await sleep(800);
out.insAfterClick = await ev(() => document.getElementById("ins-body").textContent.slice(0, 250));
out.insTitleAfterClick = await ev(() => document.getElementById("ins-title").textContent);
await browser.close();
return out;
