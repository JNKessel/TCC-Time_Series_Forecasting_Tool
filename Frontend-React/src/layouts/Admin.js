/*!

=========================================================
* Now UI Dashboard PRO React - v1.4.0
=========================================================

* Product Page: https://www.creative-tim.com/product/now-ui-dashboard-pro-react
* Copyright 2020 Creative Tim (https://www.creative-tim.com)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/
import React, { useState, useEffect } from 'react'
import { Route, Switch, Redirect } from 'react-router-dom'
// javascript plugin used to create scrollbars on windows
import PerfectScrollbar from 'perfect-scrollbar'
// react plugin for creating notifications
import NotificationAlert from 'react-notification-alert'

// core components
import AdminNavbar from 'components/Navbars/AdminNavbar.js'
import Footer from 'components/Footer/Footer.js'
import Sidebar from 'components/Sidebar/Sidebar.js'
import FixedPlugin from 'components/FixedPlugin/FixedPlugin.js'
import Dashboard from 'views/Dashboard/Dashboard.js'

import routes from 'routes.js'

var ps

function Admin(props) {
  const [sidebarMini, setSidebarMini] = useState(true)
  const [backgroundColor, setBackgroundColor] = useState('blue')

  const notificationAlert = React.createRef()
  const mainPanel = React.createRef()

  useEffect(() => {
    if (navigator.platform.indexOf('Win') > -1) {
      document.documentElement.className += ' perfect-scrollbar-on'
      document.documentElement.classList.remove('perfect-scrollbar-off')
      ps = new PerfectScrollbar(mainPanel.current)
    }

    if (props.history.action === 'PUSH') {
      document.documentElement.scrollTop = 0
      document.scrollingElement.scrollTop = 0
      mainPanel.current.scrollTop = 0
    }

    return () => {
      if (navigator.platform.indexOf('Win') > -1) {
        ps.destroy()
        document.documentElement.className += ' perfect-scrollbar-off'
        document.documentElement.classList.remove('perfect-scrollbar-on')
      }
    }
  })

  const minimizeSidebar = () => {
    var message = 'Sidebar mini '
    if (document.body.classList.contains('sidebar-mini')) {
      setSidebarMini(false)
      message += 'deactivated...'
    } else {
      setSidebarMini(true)
      message += 'activated...'
    }
    document.body.classList.toggle('sidebar-mini')
    var options = {}
    options = {
      place: 'tr',
      message: message,
      type: 'info',
      icon: 'now-ui-icons ui-1_bell-53',
      autoDismiss: 7,
    }
    notificationAlert.current.notificationAlert(options)
  }
  const handleColorClick = (color) => {
    setBackgroundColor(color)
  }
  const getRoutes = (routes) => {
    return routes.map((prop2, key) => {
      if (prop2.collapse) {
        return getRoutes(prop2.views)
      }
      if (prop2.layout === '/admin') {
        if (prop2.name === 'Dashboard') {
          return (
            <Route
              path={prop2.layout + prop2.path}
              component={() => <Dashboard />}
              key={key}
            />
          )
        }
        return (
          <Route
            path={prop2.layout + prop2.path}
            component={prop2.component}
            key={key}
          />
        )
      } else {
        return null
      }
    })
  }
  const getActiveRoute = (routes) => {
    let activeRoute = 'Default Brand Text'
    for (let i = 0; i < routes.length; i++) {
      if (routes[i].collapse) {
        let collapseActiveRoute = getActiveRoute(routes[i].views)
        if (collapseActiveRoute !== activeRoute) {
          return collapseActiveRoute
        }
      } else {
        if (
          window.location.pathname.indexOf(
            routes[i].layout + routes[i].path
          ) !== -1
        ) {
          return routes[i].name
        }
      }
    }
    return activeRoute
  }

  return (
    <div className='wrapper'>
      <NotificationAlert ref={notificationAlert} />
      {false && (
        <Sidebar
          {...props}
          routes={routes}
          minimizeSidebar={minimizeSidebar}
          backgroundColor={backgroundColor}
        />
      )}
      <div className='main-panel' ref={mainPanel}>
        <AdminNavbar {...props} brandText={getActiveRoute(routes)} />
        <Switch>
          {getRoutes(routes)}
          <Redirect from='/admin' to='/admin/dashboard' />
        </Switch>
        {
          // we don't want the Footer to be rendered on full screen maps page
          window.location.href.indexOf('full-screen-maps') !== -1 ? null : (
            <Footer fluid />
          )
        }
      </div>
      {false && (
        <FixedPlugin
          handleMiniClick={minimizeSidebar}
          sidebarMini={sidebarMini}
          bgColor={backgroundColor}
          handleColorClick={handleColorClick}
        />
      )}
    </div>
  )
}

export default Admin
