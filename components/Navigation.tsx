"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/prediction", label: "Prediction" },
  { href: "/fairness", label: "Fairness" },
  { href: "/generalization", label: "Generalization" },
  { href: "/models", label: "Models" },
  { href: "/eda", label: "EDA" },
  { href: "/diagnostic", label: "Diagnostic" },
  { href: "/about", label: "About" },
];

export function Navigation() {
  const pathname = usePathname();

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <nav style={{
      background: "var(--primary-navy)",
      color: "white",
      padding: "16px 20px",
      marginBottom: "0"
    }}>
      <div style={{
        maxWidth: "1140px",
        margin: "0 auto",
        display: "flex",
        gap: "24px",
        alignItems: "center"
      }}>
        <Link href="/" style={{
          fontSize: "18px",
          fontWeight: 600,
          textDecoration: "none",
          color: "white"
        }}>
          COMPAS Dashboard
        </Link>

        <div style={{
          display: "flex",
          gap: "8px",
          marginLeft: "auto"
        }}>
          {links.map(({ href, label }) => {
            const active = isActive(href);
            return (
              <Link
                key={href}
                href={href}
                style={{
                  textDecoration: "none",
                  fontSize: "14px",
                  fontWeight: active ? 700 : 400,
                  color: active ? "white" : "rgba(255,255,255,0.65)",
                  padding: "5px 12px",
                  borderRadius: "6px",
                  background: active ? "rgba(255,255,255,0.15)" : "transparent",
                  borderBottom: active ? "2px solid #38bdf8" : "2px solid transparent",
                  transition: "all 150ms",
                }}
              >
                {label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
