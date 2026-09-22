"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import menuStyles from "./public-mobile-menu.module.css";
import styles from "./public-marketing-shell.module.css";

type MobileNavigationItem = {
  label: string;
  href: string;
};

export function PublicMobileMenu({
  items,
  navigationLabel,
  menuLabel,
  closeLabel,
  openNavigationLabel,
  closeNavigationLabel,
}: {
  items: MobileNavigationItem[];
  navigationLabel: string;
  menuLabel: string;
  closeLabel: string;
  openNavigationLabel: string;
  closeNavigationLabel: string;
}) {
  const detailsRef = useRef<HTMLDetailsElement | null>(null);
  const summaryRef = useRef<HTMLElement | null>(null);
  const [open, setOpen] = useState(false);

  function closeMenu({ restoreFocus = false } = {}) {
    if (detailsRef.current) detailsRef.current.open = false;
    setOpen(false);
    if (restoreFocus) summaryRef.current?.focus();
  }

  useEffect(() => {
    if (!open) return;

    function handlePointerDown(event: PointerEvent) {
      const details = detailsRef.current;
      if (!details || details.contains(event.target as Node)) return;
      details.open = false;
      setOpen(false);
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [open]);

  function handleKeyDown(event: KeyboardEvent<HTMLDetailsElement>) {
    if (event.key !== "Escape" || !detailsRef.current?.open) return;
    event.preventDefault();
    closeMenu({ restoreFocus: true });
  }

  return (
    <details
      ref={detailsRef}
      className={styles.mobileMenu}
      onToggle={(event) => setOpen(event.currentTarget.open)}
      onKeyDown={handleKeyDown}
    >
      <summary
        ref={summaryRef}
        aria-expanded={open}
        aria-label={open ? closeNavigationLabel : openNavigationLabel}
      >
        <span
          aria-hidden="true"
          className={`${styles.menuIcon} ${menuStyles.menuIcon} ${open ? menuStyles.menuIconOpen : ""}`}
        >
          <i /><i /><i />
        </span>
        <span>{open ? closeLabel : menuLabel}</span>
      </summary>
      <nav aria-label={navigationLabel}>
        {items.map((item) => (
          <Link
            href={item.href}
            key={`mobile-${item.href}-${item.label}`}
            onClick={() => closeMenu()}
          >
            {item.label}
          </Link>
        ))}
      </nav>
    </details>
  );
}
