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
import React from "react";
import { Route, Switch, Redirect } from "react-router-dom";

// core components
import AuthNavbar from "components/Navbars/AuthNavbar.js";
import Footer from "components/Footer/Footer.js";
import FixedPlugin from "components/FixedPlugin/FixedPlugin.js";

import routes from "routes.js";

class Auth extends React.Component {
  state = {
    filterColor: "yellow",
  };
  getRoutes = (routes) => {
    return routes.map((prop, key) => {
      if (prop.collapse) {
        return this.getRoutes(prop.views);
      }
      if (prop.layout === "/auth") {
        return (
          <Route
            path={prop.layout + prop.path}
            component={prop.component}
            key={key}
          />
        );
      } else {
        return null;
      }
    });
  };
  handleColorClick = (color) => {
    this.setState({ filterColor: color });
  };
  render() {
    return (
      <>
        <AuthNavbar {...this.props} />
        <div className="wrapper wrapper-full-page">
          <div
            className="full-page section-image"
            filter-color={this.state.filterColor}
          >
            <Switch>
              {this.getRoutes(routes)}
              <Redirect from="/auth" to="/auth/login-page" />
            </Switch>
            <Footer fluid />
          </div>
        </div>
        <FixedPlugin
          bgColor={this.state.filterColor}
          handleColorClick={this.handleColorClick}
        />
      </>
    );
  }
}

export default Auth;
