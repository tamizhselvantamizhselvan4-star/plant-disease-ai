/* ============================================================
   MOBILE NAV FIX
   ------------------------------------------------------------
   Problem this solves:

   Every bottom-nav tab (Home / Chat / AI / Scan / Care / Health)
   is a normal <a href="..."> link, so switching tabs pushes a
   brand new entry onto the browser's history stack every time.
   Visit Home -> Scan -> Treatment -> Chat -> Care -> Health and
   the phone's back button (or a mobile "exit" gesture) has to
   walk back through all 6 of those pages one at a time instead
   of just leaving the app.

   Fix, in two parts:

   1. Clicking a bottom-nav tab REPLACES the current history
      entry (location.replace) instead of pushing a new one.
      Tabs behave like tabs - switching between them doesn't
      grow the stack. Normal in-page links (a scan result, a
      treatment record, "back to treatment", login/logout,
      etc.) are completely untouched and keep stacking/behaving
      exactly as they do today.

   2. On the Home page only, a single "press back again to
      exit" guard is added (the same pattern most native
      Android apps use on their root screen), so a single
      accidental back-press doesn't immediately kick the user
      out now that tabs no longer stack.

   Usage:
     - Add   <script src="{{ url_for('static', filename='js/mobile-nav.js') }}"></script>
       right before </body> on every page that has the bottom
       nav bar (Home, Chat/AI, Scan, Care, Health, etc.).
     - On the Home page ONLY, also add data-app-root="true" to
       the <body> tag, e.g.
         <body data-app-root="true" ...>
       This marks it as the page that should show the exit
       guard. Every other page should NOT have this attribute.
============================================================ */

(function () {
    "use strict";

    // --------------------------------------------------------
    // 1. BOTTOM-NAV TABS: REPLACE INSTEAD OF PUSH
    // --------------------------------------------------------

    function isSameOrigin(url) {
        try {
            var a = document.createElement("a");
            a.href = url;
            return a.origin === window.location.origin;
        } catch (e) {
            return false;
        }
    }

    document.addEventListener("click", function (event) {

        var link = event.target.closest(
            "a.bottom-nav-item, a.nav-item, .bottom-nav a, .nav-inner a"
        );

        if (!link) return;

        var href = link.getAttribute("href");

        if (
            !href ||
            href.charAt(0) === "#" ||
            href.indexOf("javascript:") === 0 ||
            link.target === "_blank" ||
            link.hasAttribute("download") ||
            event.defaultPrevented ||
            event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey || event.altKey
        ) {
            return;
        }

        var absoluteUrl;
        try {
            absoluteUrl = new URL(href, window.location.href).href;
        } catch (e) {
            return;
        }

        if (!isSameOrigin(absoluteUrl)) return;

        // Already on this exact page - let the default click happen
        // (or do nothing) instead of forcing a reload via replace.
        if (absoluteUrl === window.location.href) return;

        event.preventDefault();
        window.location.replace(absoluteUrl);
    }, true);


    // --------------------------------------------------------
    // 2. "PRESS BACK AGAIN TO EXIT" ON THE HOME PAGE
    // --------------------------------------------------------

    var isAppRoot = document.body.getAttribute("data-app-root") === "true";

    if (isAppRoot && window.history && window.history.pushState) {

        history.pushState({ appGuard: true }, "", window.location.href);

        var toastEl = null;
        var exitArmed = false;
        var rearmTimer = null;

        function showExitToast() {
            if (!toastEl) {
                toastEl = document.createElement("div");
                toastEl.textContent = "Press back again to exit";
                toastEl.setAttribute("role", "status");
                toastEl.style.cssText =
                    "position:fixed;left:50%;bottom:88px;transform:translateX(-50%);" +
                    "background:#163e2b;color:#fff;padding:10px 18px;border-radius:999px;" +
                    "font:600 13px Inter,'Segoe UI',Arial,sans-serif;z-index:999999;" +
                    "box-shadow:0 10px 25px rgba(0,0,0,.25);opacity:0;" +
                    "transition:opacity .2s ease;pointer-events:none;white-space:nowrap;";
                document.body.appendChild(toastEl);
            }

            toastEl.style.opacity = "1";
            clearTimeout(toastEl._hideTimer);
            toastEl._hideTimer = setTimeout(function () {
                toastEl.style.opacity = "0";
            }, 1800);
        }

        window.addEventListener("popstate", function () {

            if (exitArmed) {
                // Second back-press within the window - let this one
                // through so the browser/app actually exits/goes back
                // to whatever was open before this app.
                return;
            }

            exitArmed = true;
            showExitToast();

            // Re-plant the guard entry so the *next* back press is
            // caught again if the user doesn't actually leave.
            history.pushState({ appGuard: true }, "", window.location.href);

            clearTimeout(rearmTimer);
            rearmTimer = setTimeout(function () {
                exitArmed = false;
            }, 2000);
        });
    }
})();