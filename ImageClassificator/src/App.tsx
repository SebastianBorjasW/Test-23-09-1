import './App.css'
import AI from './Pages/AiApp'
import Stats from './Pages/Statdistics'
import React from 'react';
import { BrowserRouter as Router, Route, Routes, BrowserRouter } from 'react-router-dom';
import SidebarMenu from './components/ui/SidebarMenu';
import Statdistics from './Pages/Statdistics';


const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path ="/" >
          <Route index element={<AI />} />
        </Route>

        <Route path="/stadistics">
          <Route index element={<Statdistics />} />
        </Route>
      </Routes>
    
    </BrowserRouter>
  );
};

export default App
