import Link from "next/link";

export function Navigation() {
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
          gap: "24px",
          marginLeft: "auto"
        }}>
          <Link href="/" style={{
            textDecoration: "none",
            color: "rgba(255,255,255,0.8)",
            fontSize: "14px",
            transition: "color 200ms"
          }}>
            Home
          </Link>
          <Link href="/prediction" style={{
            textDecoration: "none",
            color: "rgba(255,255,255,0.8)",
            fontSize: "14px",
            transition: "color 200ms"
          }}>
            Prediction
          </Link>
          <Link href="/fairness" style={{
            textDecoration: "none",
            color: "rgba(255,255,255,0.8)",
            fontSize: "14px",
            transition: "color 200ms"
          }}>
            Fairness
          </Link>
          <Link href="/generalization" style={{
            textDecoration: "none",
            color: "rgba(255,255,255,0.8)",
            fontSize: "14px",
            transition: "color 200ms"
          }}>
            Generalization
          </Link>
          <Link href="/models" style={{
            textDecoration: "none",
            color: "rgba(255,255,255,0.8)",
            fontSize: "14px",
            transition: "color 200ms"
          }}>
            Models
          </Link>
          <Link href="/about" style={{
            textDecoration: "none",
            color: "rgba(255,255,255,0.8)",
            fontSize: "14px",
            transition: "color 200ms"
          }}>
            About
          </Link>
        </div>
      </div>
    </nav>
  );
}
