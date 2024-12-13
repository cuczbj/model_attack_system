// Type Imports
import type { VerticalMenuDataType } from '@/types/menuTypes'

const verticalMenuData = (): VerticalMenuDataType[] => [
  {
    label: '主页',
    href: '/home',
    icon: 'tabler-smart-home'
  },
  {
    label: '设置',
    href: '/about',
    icon: 'tabler-info-circle'
  }
]

export default verticalMenuData
