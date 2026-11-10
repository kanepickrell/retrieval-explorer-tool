import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ExplorerProvider } from "@/lib/explorer-context";
import Header from "@/components/Header";
import Explorer from "@/pages/Explorer";
import Compare from "@/pages/Compare";
import Prospector from "@/pages/Prospector";

export default function App() {
  return (
    <ExplorerProvider>
      <BrowserRouter>
        <div className="mx-auto max-w-[1440px] px-4 pb-10 sm:px-6">
          <Header />
          <Routes>
            <Route path="/" element={<Explorer />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/prospector" element={<Prospector />} />
            <Route path="*" element={<Explorer />} />
          </Routes>
        </div>
      </BrowserRouter>
    </ExplorerProvider>
  );
}
