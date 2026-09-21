export function getAuthConfirmationUrl(browserOrigin: string, configuredSiteUrl?: string) {
  const configured = configuredSiteUrl?.trim();
  const baseUrl = (configured || browserOrigin).replace(/\/+$/, "");
  return `${baseUrl}/auth/confirm`;
}
