"""Record the demo (video + screenshots) against a running server:
  python -m sagefemme.server --fresh &   then   python tools/record_demo.py
Covers: offline capture, return of connectivity, review of uncertain fields, match decision,
re-digitisation of the same registry, lifecycle view, record view, dashboard.
"""
import asyncio
import sys

from playwright.async_api import async_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "demo"
PICK = ("Correct", "Tout correct", "Cochée", "Vide sur le papier", "Passer")


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1400, "height": 860}, record_video_dir=f"{OUT}/video",
                                  record_video_size={"width": 1400, "height": 860})
        pg = await ctx.new_page()

        async def bot_click(text, timeout=180000):
            loc = pg.locator(f'.msg.bot button:not([disabled]):has-text("{text}")').last
            await loc.wait_for(timeout=timeout)
            await pg.wait_for_timeout(700)
            await loc.click()
            await pg.wait_for_timeout(1200)

        async def answer_all(fast=450):
            for i in range(150):
                last = pg.locator(".msg.bot").last
                if "Récapitulatif" in await last.inner_text():
                    return
                btns = [await x.inner_text() for x in await last.locator("button:not([disabled])").all()]
                txt = await last.inner_text()
                if not btns and ("Écrivez les valeurs" in txt or "Écrivez la valeur" in txt):
                    n = txt.split("ordre")[1].split("séparées")[0].strip().rstrip(",").count(",") + 1 if "ordre" in txt else 1
                    await pg.fill("#text", "; ".join(["?"] * n))   # mark as illegible for the demo
                    await pg.press("#text", "Enter")
                    await pg.wait_for_timeout(1200)
                    continue
                if not btns:
                    await pg.wait_for_timeout(500)
                    continue
                pick = next((x for x in PICK if any(x in y for y in btns)), btns[0])
                await last.locator(f'button:has-text("{pick}")').first.click()
                await pg.wait_for_timeout(fast if i > 3 else 1500)

        async def shot(name):
            await pg.screenshot(path=f"{OUT}/{name}.png")

        await pg.goto("http://localhost:8000/")
        await pg.wait_for_timeout(1500)
        await pg.click("#lg-go")
        await pg.wait_for_timeout(2000)
        # 1. offline capture
        await pg.uncheck("#online")
        await pg.wait_for_timeout(1200)
        await bot_click("Nouveau registre")
        await pg.select_option("#lvl", "mild")
        await pg.select_option("#pat", "P03")
        await pg.wait_for_timeout(800)
        await pg.click("#sendall")
        await pg.locator('.msg.bot:has-text("Page 8 reçue")').wait_for(timeout=120000)
        await bot_click("Terminer")
        await pg.wait_for_timeout(2500)
        await shot("1_offline_capture")
        await pg.click('#tabs button[data-p="records"]')
        await pg.wait_for_timeout(2500)
        await pg.click('#tabs button[data-p="demo"]')
        # 2. connectivity returns
        await pg.check("#online")
        await pg.locator('.msg.bot button:has-text("Vérifier maintenant")').wait_for(timeout=180000)
        await pg.wait_for_timeout(1500)
        await shot("2_back_online")
        # 3. review of uncertain fields
        await bot_click("Vérifier maintenant")
        await bot_click("essentiels")
        await pg.wait_for_timeout(2500)
        await shot("3_review_code")
        await bot_click("Corriger")
        await pg.fill("#text", "2026-112-003")
        await pg.press("#text", "Enter")
        await pg.wait_for_timeout(3000)
        await shot("4_review_field")
        await answer_all()
        await pg.wait_for_timeout(1500)
        await shot("5_summary")
        # 4. match decision
        await bot_click("Valider le dossier")
        await pg.wait_for_timeout(1500)
        await shot("6_match_none")
        await bot_click("créer")
        await pg.wait_for_timeout(4000)
        # re-digitisation of the same registry (2 pages)
        await bot_click("Nouveau registre")
        figs = await pg.query_selector_all("#thumbs figure")
        names = [(await f.get_attribute("data-n"))[4:-4] for f in figs]
        for want in ("cover", "identification"):
            await figs[names.index(want)].click()
            await pg.wait_for_timeout(3500)
        await bot_click("Terminer")
        await bot_click("Vérifier maintenant")
        m = pg.locator('.msg.bot button:not([disabled]):has-text("essentiels")')
        if await m.count():
            await m.last.click()
            await pg.wait_for_timeout(1500)
        await bot_click("Corriger")
        await pg.fill("#text", "2026-112-003")
        await pg.press("#text", "Enter")
        await pg.wait_for_timeout(2000)
        await answer_all()
        await bot_click("Valider le dossier")
        await pg.wait_for_timeout(2500)
        await shot("7_match_found")
        await bot_click("Patiente 1")
        await pg.wait_for_timeout(2500)
        await shot("8_redigitisation")
        b2 = pg.locator('.msg.bot button:not([disabled]):has-text("nouvelles infos")')
        if await b2.count():
            await b2.first.click()
            await pg.wait_for_timeout(4000)
        await pg.click('#tabs button[data-p="records"]')
        await pg.wait_for_timeout(3000)
        await shot("9_lifecycle")
        await pg.locator("#rectable button").last.click()
        await pg.wait_for_timeout(3000)
        await shot("10_record_view")
        await pg.click('#tabs button[data-p="dash"]')
        await pg.wait_for_timeout(3000)
        await shot("11_dashboard")
        await ctx.close()
        await b.close()


asyncio.run(main())
