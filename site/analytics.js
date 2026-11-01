/* Optional, consent-gated GA4 measurement. Set the web stream ID to enable. */
(() => {
  const measurementId = "";
  const consentKey = "cuda-compat-analytics-consent";
  const allowedEvents = new Set([
    "installation_guide_click",
    "support_matrix_click",
    "gpu_catalog_click",
    "project_repository_click",
    "evidence_page_click",
    "external_reference_click",
  ]);

  if (!/^G-[A-Z0-9]+$/.test(measurementId)) {
    document.querySelectorAll("[data-analytics-settings], [data-analytics-revoke]").forEach((button) => {
      button.hidden = true;
    });
    return;
  }

  const privacyPath = location.pathname.includes("/docs/") ? "privacy.html" : "docs/privacy.html";
  const banner = document.createElement("aside");
  banner.className = "analytics-banner";
  banner.hidden = true;
  banner.setAttribute("role", "region");
  banner.setAttribute("aria-labelledby", "analytics-banner-title");
  banner.innerHTML = `<div><strong id="analytics-banner-title">Optional site analytics</strong><p>Allow Google Analytics to measure page visits and clicks on key project links. No analytics tag loads unless you accept. You can change this choice from the privacy page.</p></div><div class="analytics-actions"><button type="button" data-analytics-choice="accept">Allow analytics</button><button type="button" data-analytics-choice="reject">Reject</button><a href="${privacyPath}">Details</a></div>`;
  document.body.append(banner);

  const settings = document.querySelectorAll("[data-analytics-settings]");
  const showBanner = () => {
    banner.hidden = false;
    banner.querySelector('[data-analytics-choice="accept"]').focus();
  };
  settings.forEach((button) => button.addEventListener("click", showBanner));

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

  const hasConsent = () => {
    try {
      return localStorage.getItem(consentKey) === "accepted";
    } catch {
      return false;
    }
  };

  const setConsent = (choice) => {
    try {
      localStorage.setItem(consentKey, choice);
    } catch {
      // Without persistent consent storage, keep analytics disabled.
      return;
    }
    banner.hidden = true;
    if (choice === "accepted") loadAnalytics();
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

  banner.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : event.target.parentElement;
    const button = target?.closest("[data-analytics-choice]");
    if (!button) return;
    setConsent(button.dataset.analyticsChoice === "accept" ? "accepted" : "rejected");
  });

  const classifyLink = (link) => {
    const url = new URL(link.href, location.href);
    const path = url.pathname.toLowerCase();
    if (url.hostname === "github.com" && path === "/zachshallbetter/cuda-compatibility") return "project_repository_click";
    if (path.endsWith("/getting-started.html")) return "installation_guide_click";
    if (path.endsWith("/compatibility.html")) return "support_matrix_click";
    if (path.endsWith("/gpu-catalog.html")) return "gpu_catalog_click";
    if (path.endsWith("/research.html")) return "evidence_page_click";
    if (url.origin !== location.origin && url.protocol.startsWith("http")) return "external_reference_click";
    return null;
  };

  document.addEventListener("click", (event) => {
    if (!hasConsent() || !initialized) return;
    const target = event.target instanceof Element ? event.target : event.target.parentElement;
    const link = target?.closest("a[href]");
    if (!link) return;
    const name = classifyLink(link);
    if (!allowedEvents.has(name)) return;
    const url = new URL(link.href, location.href);
    window.gtag("event", name, {
      page_path: location.pathname,
      ...(url.origin !== location.origin ? { link_domain: url.hostname } : {}),
    });
  });

  const revoke = () => setConsent("rejected");
  document.querySelectorAll("[data-analytics-revoke]").forEach((button) => button.addEventListener("click", revoke));

  let choice = "";
  try {
    choice = localStorage.getItem(consentKey) || "";
  } catch {
    choice = "";
  }
  if (choice === "accepted") loadAnalytics();
  else if (choice !== "rejected") showBanner();
})();
