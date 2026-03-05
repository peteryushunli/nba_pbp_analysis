import { Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <nav className="bg-gray-900 text-white px-6 py-3 flex items-center gap-4">
        <span className="font-bold text-lg">NBA PBP Analysis</span>
        <span className="text-sm text-gray-400">On/Off Efficiency</span>
      </nav>
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
