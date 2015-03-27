/**
 * @file component_tree_node.h
 * @brief This file contains class ComponentTreeNode, which can be 
 * used as node in the component tree.
 * @author Tran Tuan Hiep
 * @version 1.0
 * @date 2014-09-13
 */

#ifndef NOCRLIB_COMPONENT_TREE_NODE_H
#define NOCRLIB_COMPONENT_TREE_NODE_H

#include <stack>

/**
 * @brief ComponentTreeNode is an option for node in ComponentTreeBuilder
 *
 * @tparam T type T specifies the type of class, that contain 
 * neccesery information about component. 
 */
template < typename T> 
class ComponentTreeNode
{
    public:
        typedef ComponentTreeNode<T> Node;
        Node *parent_;
        Node *next_, *prev_;
        Node *child_, *last_child_;

        int depth_from_parent_;

        /**
         * @brief constructor
         *
         * @param val information about component in image
         */
        ComponentTreeNode( const T &val )
            : parent_(nullptr), next_(nullptr), prev_(nullptr), 
            child_(nullptr), depth_from_parent_(0), val_(val)
        { 
        }

        ~ComponentTreeNode()
        {
            parent_ = nullptr;
            next_ = nullptr;
            prev_ = nullptr;
            child_ = nullptr;
        }

        int size_;

        /**
         * @brief returns the information class T about the component
         *
         * @return reference to the class T
         */
        T& getVal() { return val_; } 


        /**
         * @brief append new child to the node
         *
         * @param new_child new child 
         */
        void addChild( Node * new_child )
        {
            new_child->next_ = child_;
            if( child_ )
            {
                child_->prev_ = new_child;
            }
            else
            {
                last_child_ = new_child;
            }
            child_ = new_child;
            new_child->depth_from_parent_ = 1;
            new_child->parent_ = this;
        }

        /**
         * @brief reconnects all children node from \p node
         * to this node
         *
         * @param node node, whose child we reconnect
         */
        void reconnectChildren( Node * node )
        {
            addChild( node );
            node->remove();
        }

        /**
         * @brief remove this node from his parent, and reconnect 
         * this nodes children to his parent
         */
        void remove()
        {
            for ( Node *reg = child_; reg != nullptr; reg = reg->next_ )  
            {
                reg->parent_ = parent_;
                reg->depth_from_parent_ += depth_from_parent_; 
            }
            // not last and not first node
            if ( next_ && prev_ ) 
            {
                if ( child_ )
                {
                    prev_->next_ = child_;
                    child_->prev_ = prev_;
                    next_->prev_ = last_child_;
                    last_child_->next_ = next_;
                }
                else 
                {
                    prev_->next_ = next_;
                    next_->prev_ = prev_;
                }
                return;
            }

            // last node
            if ( prev_ )
            {
                if ( child_ )
                {
                    prev_->next_ = child_;
                    child_->prev_ = prev_;
                    parent_->last_child_ = last_child_; 
                }
                else 
                {
                    prev_->next_ = nullptr;
                    parent_->last_child_ = prev_;
                }
                return;
            }

            // first node
            if ( next_ )
            {
                if ( child_ )
                {
                    parent_->child_ = child_; 
                    last_child_->next_ = next_;
                    next_->prev_ = last_child_;
                }
                else
                {
                    next_->prev_ = nullptr;
                    parent_->child_ = next_;
                }
                return;
            }

            // node is only child
            if ( !next_ && !prev_ )
            {
                parent_->child_ = child_;
                parent_->last_child_ = last_child_;
            }
        }

        template <typename Functor>
        void visit(Functor & functor)
        {
            std::stack<ComponentTreeNode<T> *> stack_node;

            for ( auto * node = child_; node != nullptr; node = node->next_ )
            {
                stack_node.push(node);
            }

            while(!stack_node.empty())
            {
                auto * top_stack = stack_node.top();
                functor(top_stack->getVal());
                stack_node.pop();

                for ( auto * node = top_stack->child_; node != nullptr; node = node->next_ )
                {
                    stack_node.push(node);
                }
            }
        }


    private:
        T val_;
};





#endif /* component_tree_node.h */
