const MainFooter = () => {
    return (
        <footer className="footer bg-dark text-white p-1">
            <div className="container-fluid d-flex justify-content-between align-items-center">
                <div>&copy; {new Date().getFullYear()} Brickfish</div>
            </div>

        </footer >
    )
}

export default MainFooter