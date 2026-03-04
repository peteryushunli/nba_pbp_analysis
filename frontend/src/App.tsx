import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Layout } from "./components/Layout";
import { EfgHeatmapPage } from "./pages/EfgHeatmapPage";
import { RankingsPage } from "./pages/RankingsPage";
import { RatingsPage } from "./pages/RatingsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<EfgHeatmapPage />} />
            <Route path="rankings" element={<RankingsPage />} />
            <Route path="ratings" element={<RatingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
