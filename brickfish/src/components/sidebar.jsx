import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const Sidebar = () => {
  return (
    <div className="d-flex flex-column flex-shrink-0 p-3 bg-light" style={{ width: '250px', height: '100vh' }}>
      <a href="/" className="d-flex align-items-center mb-3 mb-md-0 me-md-auto link-dark text-decoration-none">
        <span className="fs-4">Sidebar</span>
      </a>
      <hr />
      <ul className="nav nav-pills flex-column mb-auto">
        <li className="nav-item">
          <a href="#" className="nav-link active" aria-current="page">
            Dashboard
          </a>
        </li>
        <li>
          <a href="#" className="nav-link link-dark">
            Orders
          </a>
        </li>
        <li>
          <a href="#" className="nav-link link-dark">
            Products
          </a>
        </li>
        <li>
          <a href="#" className="nav-link link-dark">
            Customers
          </a>
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;