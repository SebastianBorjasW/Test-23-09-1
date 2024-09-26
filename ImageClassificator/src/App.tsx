import './App.css'
import AI from './Pages/AiApp'
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, BrowserRouter, useNavigate } from 'react-router-dom';
import Statdistics from './Pages/Statdistics';
import Layout from '@/layouts/Layout';
import BasicLayout from '@/layouts/BasicLayout'
import SidebarMenu from './components/ui/SidebarMenu';

const AppRoutes = () => {
  const navigate = useNavigate();

  useEffect(() => {
    if(window.location.pathname == '/'){
      navigate('/AiApp');
    }
  }, [navigate]);

  return (
    <>
      <SidebarMenu />
      <Routes>
          <Route path="/AiApp" element={<BasicLayout />}>
            <Route index element={<AI />} />
          </Route>
          <Route path="/stadistics" element={<Layout />}>
            <Route index element={<Statdistics />} />
          </Route>
      </Routes>
    </>
  );


};


const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
};

export default App
