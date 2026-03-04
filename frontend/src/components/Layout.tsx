import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/", label: "eFG% Heatmap" },
  { to: "/rankings", label: "Weighted Rankings" },
  { to: "/ratings", label: "ORtg / DRtg" },
];

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-6">
        <span className="font-bold text-lg mr-4">NBA PBP Analysis</span>
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            className={({ isActive }) =>
              `text-sm font-medium px-2 py-1 rounded ${
                isActive
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:text-gray-900"
              }`
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
