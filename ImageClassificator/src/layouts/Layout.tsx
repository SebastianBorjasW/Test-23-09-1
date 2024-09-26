import SidebarMenu from "@/components/ui/SidebarMenu";
import { Outlet } from "react-router-dom";

export default function Layout() {
    return <div className="grid min-h screen w-full md:grid-cols-[220px_1fr]">
        <SidebarMenu />
        <div className="flex flex-col">
            <main className="flex flex-1 flex-col gap-4 p-4 lg:gap-6 lg:p-6">
                <Outlet />
            </main>
        </div>
    </div>
}