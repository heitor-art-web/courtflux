export type HomeCta = {
  label: string;
  href: string;
};

export type HomeMedia = {
  src: string;
  alt: string;
  width: number;
  height: number;
};

export type HeroSection = {
  type: "hero";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  primary: HomeCta;
  secondary: HomeCta;
  pills: string[];
  media: HomeMedia;
  overlay: {
    kicker: string;
    title: string;
    metrics: string[];
    note: string;
  };
};

export type ValueStripSection = {
  type: "valueStrip";
  id?: string;
  items: Array<{
    title: string;
    copy: string;
  }>;
};

export type StorySection = {
  type: "story";
  id?: string;
  variant: "media-left" | "media-right";
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  points?: Array<{
    title: string;
    copy: string;
  }>;
  note?: string;
  media: HomeMedia;
};

export type EvidenceSection = {
  type: "evidence";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  insight: {
    kicker: string;
    title: string;
    copy: string;
    action: string;
  };
  media: HomeMedia;
};

export type AudienceSection = {
  type: "audiences";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  actionLabel: string;
  items: Array<{
    audience: string;
    title: string;
    copy: string;
    href: string;
  }>;
};

export type HowSection = {
  type: "how";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  steps: Array<{
    number: string;
    title: string;
    copy: string;
  }>;
};

export type PricingPlan = {
  kicker: string;
  name: string;
  price: string;
  suffix: string;
  annualPrice: string;
  annualSuffix: string;
  description: string;
  allowance: string;
  allowanceCopy: string;
  features: string[];
  cta: string;
  href: string;
  featured: boolean;
  tag?: string;
};

export type PricingSection = {
  type: "pricing";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  trial: {
    kicker: string;
    title: string;
    copy: string;
    meta: string;
    cta: HomeCta;
  };
  plans: PricingPlan[];
  organisation: {
    title: string;
    copy: string;
    cta: HomeCta;
  };
  note: string;
};

export type FaqSection = {
  type: "faq";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  items: Array<{
    question: string;
    answer: string;
  }>;
};

export type FinalCtaSection = {
  type: "finalCta";
  id?: string;
  eyebrow: string;
  title: string;
  highlight: string;
  copy: string;
  primary: HomeCta;
  secondary: HomeCta;
};

export type HomeSection =
  | HeroSection
  | ValueStripSection
  | StorySection
  | EvidenceSection
  | AudienceSection
  | HowSection
  | PricingSection
  | FaqSection
  | FinalCtaSection;

export type HomeContent = {
  metadata: {
    title: string;
    description: string;
    canonical: string;
    openGraph: {
      title: string;
      description: string;
      image: {
        url: string;
        width: number;
        height: number;
        alt: string;
      };
    };
    twitter: {
      title: string;
      description: string;
      image: string;
    };
  };
  schema: {
    organizationDescription: string;
    softwareDescription: string;
    featureList: string[];
    audienceType: string;
    webPageName: string;
    webPageDescription: string;
  };
  navigation: Array<{
    label: string;
    href: string;
    cta?: boolean;
  }>;
  sections: HomeSection[];
  footer: {
    meta: string;
    links: Array<{
      label: string;
      href: string;
    }>;
  };
};

export function getFaqSection(content: HomeContent): FaqSection | undefined {
  return content.sections.find((section): section is FaqSection => section.type === "faq");
}
