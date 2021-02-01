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
/*eslint-disable*/
import React from 'react'
import { Container } from 'reactstrap'
// used for making the prop types of this component
import PropTypes from 'prop-types'

class Footer extends React.Component {
  render() {
    return (
      <footer
        className={'footer' + (this.props.default ? ' footer-default' : '')}
      >
        <Container fluid={this.props.fluid}>
          <nav>
            <ul>
              <li>
                <a
                  href='https://www.creative-tim.com?ref=nudr-footer'
                  className='mr-4-px'
                  target='_blank'
                ></a>
              </li>
              <li>
                <a
                  href='https://presentation.creative-tim.com?ref=nudr-footer'
                  className='mr-4-px'
                  target='_blank'
                ></a>
              </li>
              <li>
                <a
                  href='https://blog.creative-tim.com?ref=nudr-footer'
                  target='_blank'
                ></a>
              </li>
            </ul>
          </nav>
          <div className='copyright'>
            <a href='https://www.invisionapp.com' target='_blank'></a>
            <a
              href='https://www.creative-tim.com?ref=nudr-footer'
              target='_blank'
            ></a>
          </div>
        </Container>
      </footer>
    )
  }
}

Footer.defaultProps = {
  default: false,
  fluid: false,
}

Footer.propTypes = {
  default: PropTypes.bool,
  fluid: PropTypes.bool,
}

export default Footer
