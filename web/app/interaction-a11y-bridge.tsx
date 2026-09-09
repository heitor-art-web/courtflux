"use client";

import { useEffect } from "react";

function setupRadioGroup(group: HTMLElement) {
  const radios = Array.from(group.querySelectorAll<HTMLElement>('[role="radio"]'));
  if (!radios.length) return;

  const selectedIndex = radios.findIndex((radio) => radio.getAttribute("aria-checked") === "true");
  radios.forEach((radio, index) => {
    radio.tabIndex = index === (selectedIndex >= 0 ? selectedIndex : 0) ? 0 : -1;
  });
}

function setupTabs(tab: HTMLElement) {
  const parent = tab.parentElement;
  if (!parent) return;
  if (!parent.hasAttribute("role")) parent.setAttribute("role", "tablist");

  const tabs = Array.from(parent.querySelectorAll<HTMLElement>(':scope > [role="tab"]'));
  const selectedIndex = Math.max(0, tabs.findIndex((item) => item.getAttribute("aria-selected") === "true"));
  tabs.forEach((item, index) => {
    item.tabIndex = index === selectedIndex ? 0 : -1;
    if (!item.id) item.id = `courtflux-tab-${index}`;
  });

  const activeTab = tabs[selectedIndex];
  const panel = parent.nextElementSibling;
  if (activeTab && panel instanceof HTMLElement) {
    const panelId = `courtflux-tabpanel-${selectedIndex}`;
    panel.id = panelId;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", activeTab.id);
    panel.tabIndex = 0;
    activeTab.setAttribute("aria-controls", panelId);
    tabs.forEach((item, index) => {
      if (index !== selectedIndex) item.removeAttribute("aria-controls");
    });
  }
}

function setupSourceSelector(group: HTMLElement) {
  const buttons = Array.from(group.querySelectorAll<HTMLButtonElement>("button"));
  buttons.forEach((button) => {
    const active = button.className.includes("activeSegment");
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

export function InteractionA11yBridge() {
  useEffect(() => {
    const enhance = (root: ParentNode = document) => {
      root.querySelectorAll<HTMLElement>('[role="radiogroup"]').forEach(setupRadioGroup);
      root.querySelectorAll<HTMLElement>('[role="tab"]').forEach(setupTabs);
      root.querySelectorAll<HTMLElement>('[aria-label="Video source"]').forEach(setupSourceSelector);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;

      if (target.getAttribute("role") === "radio") {
        const group = target.closest<HTMLElement>('[role="radiogroup"]');
        if (!group) return;
        const radios = Array.from(group.querySelectorAll<HTMLElement>('[role="radio"]'));
        const index = radios.indexOf(target);
        if (index < 0) return;
        let next = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % radios.length;
        else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + radios.length) % radios.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = radios.length - 1;
        else return;
        event.preventDefault();
        radios[next]?.focus();
        radios[next]?.click();
        setupRadioGroup(group);
        return;
      }

      if (target.getAttribute("role") === "tab") {
        const list = target.closest<HTMLElement>('[role="tablist"]') ?? target.parentElement;
        if (!list) return;
        const tabs = Array.from(list.querySelectorAll<HTMLElement>(':scope > [role="tab"]'));
        const index = tabs.indexOf(target);
        if (index < 0) return;
        let next = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % tabs.length;
        else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = tabs.length - 1;
        else return;
        event.preventDefault();
        tabs[next]?.focus();
        tabs[next]?.click();
        setupTabs(tabs[next]);
      }
    };

    enhance();
    document.addEventListener("keydown", onKeyDown);
    const observer = new MutationObserver(() => enhance());
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["aria-checked", "aria-selected", "class"],
    });

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      observer.disconnect();
    };
  }, []);

  return null;
}
