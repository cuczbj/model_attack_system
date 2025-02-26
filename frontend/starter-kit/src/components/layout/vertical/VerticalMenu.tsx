// MUI Imports
import { useTheme } from '@mui/material/styles'

// Third-party Imports
import PerfectScrollbar from 'react-perfect-scrollbar'

// Type Imports
import type { VerticalMenuContextProps } from '@menu/components/vertical-menu/Menu'

// Component Imports
import { Menu, MenuItem } from '@menu/vertical-menu'

// Hook Imports
import useVerticalNav from '@menu/hooks/useVerticalNav'

// Styled Component Imports
import StyledVerticalNavExpandIcon from '@menu/styles/vertical/StyledVerticalNavExpandIcon'

// Style Imports
import menuItemStyles from '@core/styles/vertical/menuItemStyles'
import menuSectionStyles from '@core/styles/vertical/menuSectionStyles'

// React Imports
import { useState } from 'react'

type RenderExpandIconProps = {
  open?: boolean
  transitionDuration?: VerticalMenuContextProps['transitionDuration']
}

type Props = {
  scrollMenu: (container: any, isPerfectScrollbar: boolean) => void
}

const RenderExpandIcon = ({ open, transitionDuration }: RenderExpandIconProps) => (
  <StyledVerticalNavExpandIcon open={open} transitionDuration={transitionDuration}>
    <i className={`tabler-chevron-${open ? 'down' : 'right'}`} />
  </StyledVerticalNavExpandIcon>
)

const VerticalMenu = ({ scrollMenu }: Props) => {
  // Hooks
  const theme = useTheme()
  const verticalNavOptions = useVerticalNav()

  // Vars
  const { isBreakpointReached, transitionDuration } = verticalNavOptions

  const ScrollWrapper = isBreakpointReached ? 'div' : PerfectScrollbar

  // Local state to handle submenu expansion
  const [isExpanded, setIsExpanded] = useState(false)

  const handleExpandClick = () => {
    setIsExpanded(!isExpanded)
  }

  return (
    <ScrollWrapper
      {...(isBreakpointReached
        ? {
            className: 'bs-full overflow-y-auto overflow-x-hidden',
            onScroll: container => scrollMenu(container, false)
          }
        : {
            options: { wheelPropagation: false, suppressScrollX: true },
            onScrollY: container => scrollMenu(container, true)
          })}
    >
      <Menu
        popoutMenuOffset={{ mainAxis: 23 }}
        menuItemStyles={menuItemStyles(verticalNavOptions, theme)}
        renderExpandIcon={({ open }) => <RenderExpandIcon open={open} transitionDuration={transitionDuration} />}
        renderExpandedMenuItemIcon={{ icon: <i className='tabler-circle text-xs' /> }}
        menuSectionStyles={menuSectionStyles(verticalNavOptions, theme)}
      >
        {/* 主页 */}
        <MenuItem href='/home' icon={<i className='tabler-smart-home' />}>
          主页
        </MenuItem>

        {/* 参数配置 */}
        <MenuItem href='/config' icon={<i className='tabler-settings' />}>
          参数配置
        </MenuItem>

        {/* 任务状态 */}
        <MenuItem href='/tasks' icon={<i className='tabler-activity' />}>
          任务状态
        </MenuItem>

        {/* 攻击结果 */}
        <MenuItem href='/results' icon={<i className='tabler-file' />}>
          攻击结果展示
        </MenuItem>



        <MenuItem
  icon={<i className='tabler-folder' />}
  onClick={handleExpandClick}
  style={{
    display: 'flex', // 使用 flex 布局
    justifyContent: 'space-between', // 文字和箭头两端对齐
    alignItems: 'center', // 垂直方向居中对齐
    width: '100%', // 确保菜单项占满可用宽度
    padding: '0 16px', // 添加左右内边距
  }}
>
  <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
   
  </span>
  <RenderExpandIcon open={isExpanded} transitionDuration={transitionDuration} />
</MenuItem>


        {/* 子菜单 */}
        {isExpanded && (
          <div
            style={{
              paddingLeft: '20px', // 子菜单缩进
              marginTop: '5px', // 子菜单与父菜单的间距
            }}
          >
            <MenuItem href='/models' icon={<i className='tabler-database' />}>
              目标模型管理
            </MenuItem>
            <MenuItem href='/methods' icon={<i className='tabler-tool' />}>
              攻击模型管理
            </MenuItem>
            <MenuItem href='/evaluation' icon={<i className='tabler-shield-check' />}>
              攻击效果评估
            </MenuItem>
                    {/* 攻击结果 */}
        <MenuItem href='/results' icon={<i className='tabler-file-text' />}>
          日志记录
        </MenuItem>
          </div>
        )}

        {/* 帮助文档 */}
        <MenuItem href='/help' icon={<i className='tabler-help-circle' />}>
          帮助文档
        </MenuItem>

        {/* 设置 */}
        <MenuItem href='/about' icon={<i className='tabler-info-circle' />}>
          设置
        </MenuItem>
      </Menu>
    </ScrollWrapper>
  )
}

export default VerticalMenu
