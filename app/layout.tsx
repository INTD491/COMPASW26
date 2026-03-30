import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";
import { Navigation } from "@/components/Navigation";

export const metadata: Metadata = {
  title: "COMPAS Dashboard - Interpretable & Fair Recidivism Prediction",
  description: "Interactive dashboard for evaluating fairness and interpretability in recidivism prediction models."
};

export default function RootLayout({
  children
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Navigation />
        {children}
      </body>
    </html>
  );
}
