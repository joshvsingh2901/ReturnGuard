import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ReturnGuard | Frozen A3 demonstrator",
  description: "A careful frontend for the frozen ReturnGuard A3 model of record.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
