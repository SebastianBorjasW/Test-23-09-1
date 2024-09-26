import './App.css'
import AI from './Pages/AiApp'
import React from 'react';
import { BrowserRouter as Router, Route, Routes, BrowserRouter } from 'react-router-dom';
import Statdistics from './Pages/Statdistics';
import Layout from '@/layouts/Layout';
import BasicLayout from '@/layouts/BasicLayout'
import SidebarMenu from './components/ui/SidebarMenu';


const App: React.FC = () => {
  return (
    <BrowserRouter>
      <SidebarMenu />
      <Routes>
          <Route path="/AiApp" element={<BasicLayout />}>
            <Route index element={<AI />} />
          </Route>
          <Route path="/stadistics" element={<Layout />}>
            <Route index element={<Statdistics />} />
          </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App
