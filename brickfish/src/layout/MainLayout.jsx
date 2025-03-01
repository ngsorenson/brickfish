import React, { useEffect } from 'react'
import MainNav from '../pages/MainNav'
import MainFooter from '../pages/MainFooter'

const MainLayout = ({ children, title }) => {
    useEffect(() => {
        if (title) {
            document.title = title;
        }
    }, [title]);
    return (
        <>
            <div className="d-flex flex-column vh-100">

                <main className="d-flex flex-column flex-grow-1 justify-content-center align-items-center">

                    {children}

                </main>


            </div >
        </>
    )
}

export default MainLayout