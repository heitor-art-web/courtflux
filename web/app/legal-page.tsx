import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";
import styles from "./legal.module.css";

export function LegalPage({
  eyebrow,
  title,
  intro,
  notice,
  children,
}: {
  eyebrow: string;
  title: string;
  intro: string;
  notice?: ReactNode;
  children: ReactNode;
}) {
  return (
    <main className={styles.legalPage}>
      <header className={styles.header}>
        <Link className={styles.brand} href="/" aria-label="CourtFlux home">
          <Image
            src="/brand/horizontal_logo_safe_crop.png"
            alt="CourtFlux — Basketball Intelligence"
            width={660}
            height={185}
            priority
          />
        </Link>
        <nav className={styles.nav} aria-label="Trust and legal navigation">
          <Link href="/trust">Trust</Link>
          <Link href="/video-data">Video & data</Link>
          <Link href="/ai-limitations">AI limitations</Link>
          <Link href="/privacy">Privacy</Link>
          <Link href="/terms">Terms</Link>
        </nav>
      </header>

      <div className={styles.content}>
        <section className={styles.hero}>
          <div className={styles.eyebrow}>{eyebrow}</div>
          <h1>{title}</h1>
          <p>{intro}</p>
          {notice ? <div className={styles.notice}>{notice}</div> : null}
        </section>
        <div className={styles.sections}>{children}</div>
      </div>

      <footer className={styles.footer}>
        <span>© 2026 CourtFlux · Private beta</span>
        <div className={styles.footerLinks}>
          <Link href="/">Home</Link>
          <Link href="/trust">Trust</Link>
          <Link href="/privacy">Privacy</Link>
          <Link href="/terms">Terms</Link>
          <Link href="/login">Sign in</Link>
        </div>
      </footer>
    </main>
  );
}

export { styles as legalStyles };
