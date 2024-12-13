// Type Imports
import type { HorizontalMenuDataType } from '@/types/menuTypes'

const horizontalMenuData = (): HorizontalMenuDataType[] => [
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

export default horizontalMenuData
