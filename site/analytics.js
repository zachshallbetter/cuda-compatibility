/* GA4 measurement without a prompt. It never loads for a browser that sends Global Privacy Control or
   Do Not Track, or that opted out on the privacy page. Set the web stream ID to enable. */
(() => {
  const measurementId = "G-Q7C82JK9GM";
  const consentKey = "cuda-compat-analytics-consent";
  const allowedEvents = new Set([
    "installation_guide_click",
    "support_matrix_click",
    "gpu_catalog_click",
    "project_repository_click",
    "evidence_page_click",
    "discovery_page_click",
    "external_reference_click",
    "source_download_click",
    "copy_command",
    "hero_replay",
  ]);

  if (!/^G-[A-Z0-9]+$/.test(measurementId)) {
    document.querySelectorAll("[data-analytics-settings]").forEach((button) => {
      button.hidden = true;
    });
    return;
  }

  const optedOut = () => {
    if (navigator.globalPrivacyControl === true || navigator.doNotTrack === "1") return true;
    try {
      return localStorage.getItem(consentKey) === "rejected";
    } catch {
      return false;
    }
  };

  const settings = document.querySelectorAll("[data-analytics-settings]");
  const labelSettings = () => {
    const signal = navigator.globalPrivacyControl === true || navigator.doNotTrack === "1";
    settings.forEach((button) => {
      button.disabled = signal;
      button.textContent = signal
        ? "Analytics is off: your browser asks sites not to track"
        : optedOut() ? "Turn analytics back on in this browser" : "Turn off analytics in this browser";
    });
  };

  let initialized = false;
  const loadAnalytics = () => {
    if (initialized) return;
    initialized = true;
    window.dataLayer = window.dataLayer || [];
    window.gtag = function gtag() { window.dataLayer.push(arguments); };
    window.gtag("consent", "default", {
      analytics_storage: "granted",
      ad_storage: "denied",
      ad_user_data: "denied",
      ad_personalization: "denied",
    });
    window.gtag("js", new Date());
    let referrerOrigin = "";
    try {
      const referrer = new URL(document.referrer);
      if (referrer.protocol === "http:" || referrer.protocol === "https:") referrerOrigin = referrer.origin;
    } catch {
      referrerOrigin = "";
    }
    window.gtag("config", measurementId, {
      send_page_view: false,
      allow_google_signals: false,
      allow_ad_personalization_signals: false,
      url_passthrough: false,
      page_location: `${location.origin}${location.pathname}`,
      page_referrer: referrerOrigin,
      page_title: document.title,
    });
    const tag = document.createElement("script");
    tag.async = true;
    tag.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
    document.head.append(tag);
    window.gtag("event", "page_view", {
      page_location: `${location.origin}${location.pathname}`,
      page_path: location.pathname,
      page_referrer: referrerOrigin,
      page_title: document.title,
    });
  };

  const setOptOut = (off) => {
    try {
      if (off) localStorage.setItem(consentKey, "rejected");
      else localStorage.removeItem(consentKey);
    } catch {
      // Without browser storage an opt-out can't be remembered, so keep analytics off for this page.
      if (!off) return;
    }
    labelSettings();
    if (!off) loadAnalytics();
    else if (initialized) {
      window.gtag("consent", "update", {
        analytics_storage: "denied",
        ad_storage: "denied",
        ad_user_data: "denied",
        ad_personalization: "denied",
      });
      for (const part of document.cookie.split(";")) {
        const name = part.split("=")[0].trim();
        if (name === "_ga" || name === "_gid" || name.startsWith("_ga_") || name.startsWith("_gat")) {
          document.cookie = `${name}=; Max-Age=0; path=/; SameSite=Lax`;
          document.cookie = `${name}=; Max-Age=0; path=/; domain=.${location.hostname}; SameSite=Lax`;
        }
      }
      location.reload();
    }
  };

  settings.forEach((button) => button.addEventListener("click", () => setOptOut(!optedOut())));

  const classifyLink = (link) => {
    const url = new URL(link.href, location.href);
    const path = url.pathname.toLowerCase();
    if (url.hostname === "github.com" && /^\/zachshallbetter\/cuda-compatibility\/(archive|releases)(\/|$)/.test(path)) return "source_download_click";
    if (url.hostname === "github.com" && path === "/zachshallbetter/cuda-compatibility") return "project_repository_click";
    if (path.endsWith("/getting-started.html")) return "installation_guide_click";
    if (path.endsWith("/compatibility.html")) return "support_matrix_click";
    if (path.endsWith("/gpu-catalog.html")) return "gpu_catalog_click";
    if (path.endsWith("/research.html")) return "evidence_page_click";
    if (path.endsWith("/discovery.html")) return "discovery_page_click";
    if (url.origin !== location.origin && url.protocol.startsWith("http")) return "external_reference_click";
    return null;
  };

  // Components report through this; it drops anything not allowlisted, sent before loading, or after an opt-out.
  const track = (name, params = {}) => {
    if (optedOut() || !initialized || !allowedEvents.has(name)) return;
    window.gtag("event", name, { page_path: location.pathname, ...params });
  };
  window.cudaCompatAnalytics = { track };

  const linkArea = (link) => {
    if (link.closest("header")) return "header";
    if (link.closest("footer")) return "footer";
    if (link.closest('nav[aria-label="Documentation"]')) return "sidebar";
    return "content";
  };

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : event.target.parentElement;
    const link = target?.closest("a[href]");
    if (!link) return;
    const name = classifyLink(link);
    if (!name) return;
    const url = new URL(link.href, location.href);
    track(name, {
      link_area: linkArea(link),
      ...(url.origin !== location.origin ? { link_domain: url.hostname } : {}),
    });
  });

  labelSettings();
  if (!optedOut()) loadAnalytics();
})();
