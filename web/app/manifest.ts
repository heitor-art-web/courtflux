import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "CourtFlux — Basketball Intelligence",
    short_name: "CourtFlux",
    description:
      "Basketball video analysis and intelligence for coaches, teams and academies.",
    start_url: "/",
    display: "standalone",
    background_color: "#030a13",
    theme_color: "#06111f",
    icons: [
      {
        src: "/brand/courtflux_app_icon_256px.png",
        sizes: "256x256",
        type: "image/png",
      },
      {
        src: "/brand/courtflux_app_icon_512px.png",
        sizes: "512x512",
        type: "image/png",
      },
    ],
  };
}
