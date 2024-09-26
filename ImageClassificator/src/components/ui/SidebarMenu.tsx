import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Menu, X, Home, Info} from 'lucide-react';
import { Button } from "@/components/ui/button"



const LINKS = [
    {
        title: "App",
        icon: Home,
        to: "/AiApp"
    },
    {
        title: "Estadisticas",
        icon: Info,
        to: "/stadistics"
    },
];



export default function SidebarMenu() {
    const location = useLocation();
    const navigate = useNavigate();
    const [isOpen, setIsOpen] = useState(false);
    
    const toggleMenu = () => {
        setIsOpen(!isOpen);
    }

    useEffect(() => {
        setIsOpen(false);
    
    }, [location]);

    return(
        <div className='relative'>
        <Button onClick={toggleMenu}
                className='fixed top-4 left-4 z-20 p-2 bg-gray-800 text-white rounded-md'>
                {isOpen ? <X size={24} /> : <Menu size={24} />}
        </Button>

        <div className={`fixed top-0 left-0 h-full w-64 bg-gray-800 text-white p-4 transform transition-transform duration-300 ease-in-out ${
      isOpen ? 'translate-x-0' : '-translate-x-full'
    }`}>
            <nav className='mt-16'>
                <ul className='space-y-4'>
                    {LINKS.map((link, index) => (
                        <li key={index}>
                            <Link 
                                to={link.to}
                                className={`flex items-center gap-3 rounded-lg px-3 py-2 transition-all ${
                                    location.pathname === link.to.split("/")[1]
                                      ? "bg-muted text-primary"
                                      : "text-muted-foreground"
                                  }`}
                            >
                                <link.icon className='h-4 w-4' />
                                {link.title}
                            </Link>
                        </li>
                    ))}
                </ul>
            </nav>
        </div>
    </div>
    )
}